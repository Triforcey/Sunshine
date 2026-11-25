/**
 * @file platform/virtualhid_pen_smoothing.h
 * @brief Paced release buffer for pen tablet events.
 */
#pragma once

// standard includes
#include <algorithm>
#include <chrono>
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
      if (last_arrival_) {
        const auto delta = std::chrono::duration<double, std::nano>(now - *last_arrival_).count();
        const auto clamped = std::clamp(delta, kMinIntervalNanos, kMaxIntervalNanos);
        const auto average = std::chrono::duration<double, std::nano>(interval_ewma_).count();
        const auto next = (kEwmaAlpha * clamped) + ((1.0 - kEwmaAlpha) * average);
        interval_ewma_ = std::chrono::nanoseconds {static_cast<std::int64_t>(next)};
      }
      last_arrival_ = now;
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
    time_point_t last_release_;  ///< Previous release time for pacing.
    std::jthread thread_;  ///< Drain thread performing paced releases.
  };
}  // namespace platf::virtualhid
