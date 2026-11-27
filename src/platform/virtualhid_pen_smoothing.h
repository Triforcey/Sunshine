/**
 * @file platform/virtualhid_pen_smoothing.h
 * @brief Paced release buffer for pen tablet events.
 */
#pragma once

// standard includes
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>

// local includes
#include "src/logging.h"
#include "src/platform/common.h"

namespace platf::virtualhid {
  /**
   * @brief Jitter buffer that releases pen events with even pacing.
   *
   * Network delivery bunches pen samples together, which the compositor
   * and drawing applications then observe as angular strokes. This buffer
   * preserves arrival order while re-pacing releases: samples are handed
   * to the release callback no faster than the exponential moving average
   * of their inter-arrival intervals, so bursts are stretched back to the
   * client's long-run cadence. A depth valve bounds the added latency: once the queue grows past a small depth, the overflow drains without pacing so the buffer cannot lag the client without bound.
   *
   * The clock is injectable for unit tests; production uses
   * std::chrono::steady_clock. Tests advancing an injected clock must
   * allow the real-time condition variable deadline to elapse or trigger
   * the same wakeups with near-realistic timestamps.
   */
  class pen_smoothing_buffer_t {
  public:
    /**
     * @brief Signature of the clock used for pacing decisions.
     */
    using clock_fn_t = std::function<std::chrono::steady_clock::time_point()>;

    /**
     * @brief Signature of the callback receiving paced pen samples.
     */
    using release_fn_t = std::function<void(const touch_port_t &, const pen_input_t &)>;

    /**
     * @brief Monotonic timestamp type used for pacing decisions.
     */
    using time_point_t = std::chrono::steady_clock::time_point;

    /**
     * @brief Construct the buffer and start its drain thread.
     *
     * @param release Callback invoked with each released sample, in order.
     * @param clock Optional injectable clock; defaults to the steady clock.
     */
    explicit pen_smoothing_buffer_t(release_fn_t release, clock_fn_t clock = {}):
        release_ {std::move(release)},
        clock_ {clock ? std::move(clock) : [] { return std::chrono::steady_clock::now(); }},
        last_release_ {clock_() - kDefaultInterval},
        thread_ {[this](std::stop_token stop_token) { drain(std::move(stop_token)); }} {}

    /**
     * @brief Flush pending samples and join the drain thread.
     *
     * Pending events are released immediately so that terminal state such
     * as a pen tip-up is never lost at session teardown.
     */
    ~pen_smoothing_buffer_t() {
      flush();
      thread_.request_stop();
      cv_.notify_all();
      thread_.join();
      BOOST_LOG(info) << "pen smoothing session: " << pushed_count_ << " samples received, "
                      << synthesized_count_ << " reconstructed across " << gap_count_
                      << " gaps, " << valve_overflow_count_ << " depth-valve overflows";
    }

    pen_smoothing_buffer_t(const pen_smoothing_buffer_t &) = delete;
    pen_smoothing_buffer_t &operator=(const pen_smoothing_buffer_t &) = delete;
    pen_smoothing_buffer_t(pen_smoothing_buffer_t &&) = delete;
    pen_smoothing_buffer_t &operator=(pen_smoothing_buffer_t &&) = delete;

    /**
     * @brief Queue one pen sample for paced release.
     *
     * The inter-arrival interval feeds the pacing average. This call is
     * safe to invoke from any thread, including the network input thread.
     *
     * @param touch_port Snapshot of the touch port the event maps to.
     * @param pen Pen event data.
     */
    void push(const touch_port_t &touch_port, const pen_input_t &pen) {
      const auto now = clock_();
      std::lock_guard lock {mtx_};

      synthesize_missing_samples(touch_port, pen);

      if (last_arrival_) {
        const auto delta = std::chrono::duration<double, std::nano>(now - *last_arrival_).count();
        const auto clamped = std::clamp(delta, kMinIntervalNanos, kMaxIntervalNanos);
        const auto average = std::chrono::duration<double, std::nano>(interval_ewma_).count();
        const auto next = (kEwmaAlpha * clamped) + ((1.0 - kEwmaAlpha) * average);
        interval_ewma_ = std::chrono::nanoseconds {static_cast<std::int64_t>(next)};
      }
      last_arrival_ = now;
      remember_timestamp_history(pen);
      ++pushed_count_;
      queue_.push_back({touch_port, pen, now});
      cv_.notify_all();
    }

    /**
     * @brief Release every queued sample immediately.
     *
     * Pacing resumes from the flush time so the next sample is not
     * artificially delayed.
     */
    void flush() {
      std::unique_lock lock {mtx_};
      std::deque<sample_t> pending;
      pending.swap(queue_);
      lock.unlock();

      for (const auto &sample : pending) {
        release_(sample.touch_port, sample.pen);
      }

      lock.lock();
      last_release_ = clock_();
    }

    /**
     * @brief Return the number of queued, unreleased samples.
     *
     * @return Current backlog size.
     */
    [[nodiscard]] std::size_t
    size() const {
      std::lock_guard lock {mtx_};
      return queue_.size();
    }

  private:
    /**
     * @brief Queued pen sample with its arrival timestamp.
     */
    struct sample_t {
      touch_port_t touch_port;  ///< Touch port snapshot at arrival time.
      pen_input_t pen;  ///< Pen event data.
      time_point_t arrival;  ///< Arrival timestamp used by the backlog valve.
    };

    /**
     * @brief Paced release loop executed on the drain thread.
     *
     * @param stop_token Stop token from the owning std::jthread.
     */
    void drain(std::stop_token stop_token) {
      std::unique_lock lock {mtx_};
      while (!stop_token.stop_requested()) {
        if (queue_.empty()) {
          cv_.wait(lock, stop_token, [this] { return !queue_.empty(); });
          continue;
        }

        const auto now = clock_();
        const auto earliest = last_release_ + interval_ewma_;
        // Depth valve: pace only while the queue fits the pacing window;
        // drain overflow without pacing so added latency stays bounded.
        if (queue_.size() >= kMaxQueuedDepth && now < earliest) {
          ++valve_overflow_count_;
          BOOST_LOG(debug) << "pen smoothing: depth valve draining " << queue_.size()
                           << " queued samples unpaced";
        }
        if (queue_.size() < kMaxQueuedDepth && now < earliest) {
          cv_.wait_until(lock, earliest);
          continue;
        }

        auto sample = queue_.front();
        queue_.pop_front();
        lock.unlock();
        release_(sample.touch_port, sample.pen);
        lock.lock();
        last_release_ = clock_();
      }
    }

    /**
     * @brief Whether an event type carries only positional updates.
     *
     * Mirrors LI_TOUCH_EVENT_HOVER (0x00) and LI_TOUCH_EVENT_MOVE (0x03)
     * from moonlight-common-c; kept private so this header stays
     * self-contained for unit tests.
     *
     * @param event_type Moonlight pen event type.
     * @return True when the event only moves the pen.
     */
    static bool
    is_positional_event(std::uint8_t event_type) {
      return event_type == kHoverEvent || event_type == kMoveEvent;
    }

    /**
     * @brief Update the client cadence average with an in-cadence interval.
     *
     * @param delta_us Interval between consecutive client timestamps, in microseconds.
     */
    void
    update_client_cadence(double delta_us) {
      const auto clamped = std::clamp(delta_us, kMinClientCadenceUs, kMaxClientCadenceUs);
      client_cadence_us_ = client_cadence_seeded_
          ? (kClientEwmaAlpha * clamped) + ((1.0 - kClientEwmaAlpha) * client_cadence_us_)
          : clamped;
      client_cadence_seeded_ = true;
    }

    /**
     * @brief Reconstruct pen samples lost in transit between timestamped samples.
     *
     * Input travels over unacknowledged UDP, so congestion drops position
     * samples; the drawing application then chords across the gap. When the
     * client session-clock gap between consecutive timestamped samples
     * exceeds the learned cadence, synthesize the missing positions with a
     * cubic Hermite spline through the known neighborhood so the stroke
     * curves instead of chords. Only interpolates within runs of positional
     * events with unchanged buttons and tool type; a gap never crosses a
     * state change.
     *
     * @param touch_port Touch port snapshot for the incoming sample.
     * @param pen The newly arrived pen sample.
     */
    void
    synthesize_missing_samples(const touch_port_t &touch_port, const pen_input_t &pen) {
      if (pen.clientTimestamp == 0 || !client_prev_pen_ || !is_positional_event(client_prev_pen_->eventType) ||
          !is_positional_event(pen.eventType) || client_prev_pen_->penButtons != pen.penButtons ||
          client_prev_pen_->toolType != pen.toolType) {
        return;
      }

      // Wrap-aware interval on the 32-bit client session clock.
      const auto delta_us = static_cast<double>(pen.clientTimestamp - client_prev_ts_);

      if (!client_cadence_seeded_) {
        // Seed the cadence from the first observed interval so a steady
        // client cadence can never read as perpetual loss: the gap test
        // measures against the learned baseline, and with acquisition
        // stamps the baseline is the true delivery rate. Until a client
        // has delivered two samples there is no baseline to measure a
        // gap against.
        update_client_cadence(delta_us);
        return;
      }

      if (delta_us <= kGapCadenceFactor * client_cadence_us_) {
        update_client_cadence(delta_us);
        return;
      }

      // Deltas far beyond the learned cadence are discontinuities, not
      // loss: client clock-domain switches or session starts jump stamps
      // by whole seconds (mixed-stamping builds produced wrap-spanning
      // ~2^32 us deltas at every contact/hover transition), and
      // interpolating across them fabricates geometry. Pass through.
      if (delta_us > kMaxReconstructableGapUs) {
        // Discontinuity: reset the gap baseline so the offset-recovery
        // pair after a clock or event-source switch re-seeds the cadence
        // instead of reading as a plausible multi-frame gap. Clock
        // artifacts (backwards wrap-spanning stamp jumps at hover/contact
        // transitions) read as billions of microseconds; genuine forward
        // stalls measured live during sustained drawing reach 116-265 ms
        // and stay far below the bound, so they reconstruct above.
        BOOST_LOG(info) << "pen smoothing: " << static_cast<std::uint64_t>(delta_us)
                        << " us stamp jump treated as discontinuity, baseline reset";
        client_cadence_seeded_ = false;
        return;
      }

      const auto missing = static_cast<std::size_t>(std::lround(delta_us / client_cadence_us_)) - 1;
      if (missing == 0) {
        return;
      }
      const auto count = std::min(missing, kMaxSynthesizedPerGap);

      BOOST_LOG(info) << "pen smoothing: reconstructing " << count << " of " << missing
                      << " samples across a " << static_cast<std::uint64_t>(delta_us) << " us gap (cadence "
                      << client_cadence_us_ << " us)";

      const auto &before = client_before_pen_ ? *client_before_pen_ : *client_prev_pen_;
      const auto &left = *client_prev_pen_;
      const auto &right = pen;

      for (std::size_t i = 1; i <= count; ++i) {
        const auto t = static_cast<double>(i) / static_cast<double>(count + 1);
        pen_input_t fill = left;
        fill.x = static_cast<float>(hermite(before.x, left.x, right.x, t));
        fill.y = static_cast<float>(hermite(before.y, left.y, right.y, t));
        fill.pressureOrDistance = static_cast<float>(lerp(left.pressureOrDistance, right.pressureOrDistance, t));
        fill.contactAreaMajor = static_cast<float>(lerp(left.contactAreaMajor, right.contactAreaMajor, t));
        fill.contactAreaMinor = static_cast<float>(lerp(left.contactAreaMinor, right.contactAreaMinor, t));
        fill.rotation = static_cast<std::uint16_t>(std::lround(lerp(left.rotation, right.rotation, t)));
        fill.tilt = static_cast<std::uint8_t>(std::lround(lerp(left.tilt, right.tilt, t)));
        fill.clientTimestamp = client_prev_ts_ + static_cast<std::uint32_t>(delta_us * t);
        queue_.push_back({touch_port, fill, clock_()});
        ++synthesized_count_;
      }
      ++gap_count_;
    }

    /**
     * @brief Remember the previous timestamped sample for gap reconstruction.
     *
     * @param pen The pen sample just queued.
     */
    void
    remember_timestamp_history(const pen_input_t &pen) {
      if (pen.clientTimestamp == 0) {
        return; // Legacy packets carry no session clock; leave history intact.
      }
      if (client_prev_pen_) {
        client_before_pen_ = client_prev_pen_;
      }
      client_prev_pen_ = pen;
      client_prev_ts_ = pen.clientTimestamp;
    }

    /**
     * @brief Cubic Hermite interpolation for loss reconstruction.
     *
     * The left tangent uses the sample before the gap's left neighbor when
     * available (Catmull-Rom form); the right tangent is clamped to the
     * chord.
     *
     * @param before Coordinate before the gap's left neighbor.
     * @param left Left neighbor of the gap.
     * @param right Right neighbor of the gap.
     * @param t Normalized position within the gap, in [0, 1].
     * @return The interpolated coordinate.
     */
    static double
    hermite(double before, double left, double right, double t) {
      const double m_left = (right - before) * 0.5;
      const double m_right = (right - left) * 0.5;
      const double t2 = t * t;
      const double t3 = t2 * t;
      const double h00 = (2 * t3) - (3 * t2) + 1;
      const double h10 = t3 - (2 * t2) + t;
      const double h01 = (-2 * t3) + (3 * t2);
      const double h11 = t3 - t2;
      return (h00 * left) + (h10 * m_left) + (h01 * right) + (h11 * m_right);
    }

    /**
     * @brief Linear interpolation helper.
     *
     * @param a Start value.
     * @param b End value.
     * @param t Normalized position, in [0, 1].
     * @return The interpolated value.
     */
    static double
    lerp(double a, double b, double t) {
      return a + ((b - a) * t);
    }

    static constexpr std::uint8_t kHoverEvent = 0x00;  ///< LI_TOUCH_EVENT_HOVER, mirrored from moonlight-common-c.
    static constexpr std::uint8_t kMoveEvent = 0x03;  ///< LI_TOUCH_EVENT_MOVE, mirrored from moonlight-common-c.
    static constexpr double kClientEwmaAlpha = 0.1;  ///< Weight of the newest interval in the client cadence average.
    static constexpr double kGapCadenceFactor = 2.0;  ///< Client-clock gap beyond this multiple of cadence reads as loss.
    static constexpr double kMinClientCadenceUs = 1'000.0;  ///< Client cadence floor: 1 ms.
    static constexpr double kMaxClientCadenceUs = 100'000.0;  ///< Client cadence ceiling: 100 ms.
    static constexpr double kInitialClientCadenceUs = 1'500.0;  ///< Client cadence prior to the first stamped interval.
    static constexpr std::size_t kMaxSynthesizedPerGap = 16;  ///< Reconstructed samples per gap; wider gaps fill partially.
    static constexpr double kMaxReconstructableGapUs = 500'000.0;  ///< Forward stalls up to 500 ms reconstruct (cap-bounded); beyond that, discontinuity.
    static constexpr double kEwmaAlpha = 0.1;  ///< Weight of the newest interval in the pacing average.
    static constexpr double kMinIntervalNanos = 1.0e6;  ///< Pacing interval floor: 1 ms.
    static constexpr double kMaxIntervalNanos = 1.0e8;  ///< Pacing interval ceiling: 100 ms.
    static constexpr std::chrono::nanoseconds kDefaultInterval {16'000'000};  ///< Initial pacing interval.
    static constexpr std::size_t kMaxQueuedDepth = 8;  ///< Valve: queue depth beyond which overflow drains unpaced.

    release_fn_t release_;  ///< Callback receiving paced samples.
    clock_fn_t clock_;  ///< Pacing clock, injectable for tests.
    mutable std::mutex mtx_;  ///< Guards the queue and pacing state.
    std::condition_variable_any cv_;  ///< Signals queue updates and stop requests.
    std::deque<sample_t> queue_;  ///< Pending samples in arrival order.
    std::chrono::nanoseconds interval_ewma_ {kDefaultInterval};  ///< Pacing interval average.
    std::optional<time_point_t> last_arrival_;  ///< Previous push time for the average.
    std::optional<pen_input_t> client_prev_pen_;  ///< Previous timestamped sample, for gap reconstruction.
    std::optional<pen_input_t> client_before_pen_;  ///< Sample before that, for spline tangents.
    std::uint32_t client_prev_ts_ = 0;  ///< Previous sample's client session-clock timestamp.
    double client_cadence_us_ = kInitialClientCadenceUs;  ///< EWMA of stamped client intervals, in microseconds.
    bool client_cadence_seeded_ = false;  ///< Whether the cadence average has observed an interval.
    std::uint64_t pushed_count_ = 0;  ///< Samples queued from the client (session lifetime).
    std::uint64_t synthesized_count_ = 0;  ///< Samples reconstructed across loss gaps (session lifetime).
    std::uint64_t gap_count_ = 0;  ///< Timestamp gaps detected (session lifetime).
    std::uint64_t valve_overflow_count_ = 0;  ///< Times the depth valve drained overflow unpaced (session lifetime).
    time_point_t last_release_;  ///< Previous release time for pacing.
    std::jthread thread_;  ///< Drain thread performing paced releases.
  };
}  // namespace platf::virtualhid
