/**
 * @file tests/integration/test_pen_smoothing.cpp
 * @brief Unit tests for the pen smoothing jitter buffer.
 */

// test includes
#include "../tests_common.h"

// standard includes
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

// local includes
#include "src/platform/virtualhid_pen_smoothing.h"

namespace {

  using platf::pen_input_t;
  using platf::virtualhid::pen_smoothing_buffer_t;
  using platf::touch_port_t;

  /**
   * @brief Recorded pen sample with its release timestamp.
   */
  struct recorded_sample_t {
    pen_input_t pen;  ///< The released pen sample.
    std::chrono::steady_clock::time_point released;  ///< Release timestamp.
  };

  /**
   * @brief Thread-safe recorder for release callbacks.
   */
  struct release_recorder_t {
    std::mutex mtx;  ///< Guards the recorded vector.
    std::vector<recorded_sample_t> samples;  ///< Released samples in release order.

    /**
     * @brief Build a release callback bound to this recorder.
     *
     * @return Callable suitable for the smoothing buffer.
     */
    pen_smoothing_buffer_t::release_fn_t
    callback() {
      return [this](const touch_port_t &, const pen_input_t &pen) {
        const auto now = std::chrono::steady_clock::now();
        std::lock_guard lock {mtx};
        samples.push_back({pen, now});
      };
    }

    /**
     * @brief Wait until at least the given number of samples were released.
     *
     * @param count Expected release count.
     * @param timeout Maximum time to wait.
     * @return Number of released samples when the wait ended.
     */
    std::size_t
    wait_for(std::size_t count, std::chrono::milliseconds timeout) {
      const auto deadline = std::chrono::steady_clock::now() + timeout;
      while (std::chrono::steady_clock::now() < deadline) {
        {
          std::lock_guard lock {mtx};
          if (samples.size() >= count) {
            return samples.size();
          }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      }
      std::lock_guard lock {mtx};
      return samples.size();
    }

    /**
     * @brief Snapshot the released samples under lock.
     *
     * @return Copy of the recorded samples.
     */
    std::vector<recorded_sample_t>
    snapshot() {
      std::lock_guard lock {mtx};
      return samples;
    }
  };

  /**
   * @brief Build a pen sample with an identifying coordinate.
   *
   * @param x Horizontal coordinate.
   * @param y Vertical coordinate.
   * @return The pen sample.
   */
  pen_input_t
  make_pen(float x, float y) {
    pen_input_t pen {};
    pen.x = x;
    pen.y = y;
    return pen;
  }

}  // namespace

TEST(PenSmoothingBufferTest, PassesSingleSampleThrough) {
  release_recorder_t recorder;
  {
    pen_smoothing_buffer_t buffer {recorder.callback()};
    buffer.push(touch_port_t {}, make_pen(0.25F, 0.75F));
    ASSERT_GE(recorder.wait_for(1, std::chrono::milliseconds(500)), std::size_t {1});
  }

  const auto samples = recorder.snapshot();
  ASSERT_EQ(samples.size(), std::size_t {1});
  EXPECT_FLOAT_EQ(samples.front().pen.x, 0.25F);
  EXPECT_FLOAT_EQ(samples.front().pen.y, 0.75F);
}

TEST(PenSmoothingBufferTest, PreservesArrivalOrder) {
  release_recorder_t recorder;
  {
    pen_smoothing_buffer_t buffer {recorder.callback()};
    for (auto i = 0; i < 5; ++i) {
      buffer.push(touch_port_t {}, make_pen(static_cast<float>(i), 0.0F));
    }
    ASSERT_GE(recorder.wait_for(5, std::chrono::milliseconds(500)), std::size_t {5});
  }

  const auto samples = recorder.snapshot();
  ASSERT_EQ(samples.size(), std::size_t {5});
  for (auto i = std::size_t {0}; i < samples.size(); ++i) {
    EXPECT_FLOAT_EQ(samples[i].pen.x, static_cast<float>(i));
  }
}

TEST(PenSmoothingBufferTest, StretchesBurstInsteadOfDumpingIt) {
  release_recorder_t recorder;
  {
    pen_smoothing_buffer_t buffer {recorder.callback()};
    for (auto i = 0; i < 6; ++i) {
      buffer.push(touch_port_t {}, make_pen(static_cast<float>(i), 0.5F));
    }
    ASSERT_GE(recorder.wait_for(6, std::chrono::milliseconds(500)), std::size_t {6});
  }

  const auto samples = recorder.snapshot();
  ASSERT_EQ(samples.size(), std::size_t {6});

  // Pacing must space releases apart instead of dumping the burst in one
  // go. Back-to-back delivery would show gaps of a few microseconds; the
  // pacing floor keeps them above that by an order of magnitude.
  auto min_gap = std::chrono::steady_clock::duration::max();
  for (auto i = std::size_t {1}; i < samples.size(); ++i) {
    min_gap = std::min(min_gap, samples[i].released - samples[i - 1].released);
  }
  EXPECT_GT(std::chrono::duration_cast<std::chrono::microseconds>(min_gap).count(), std::int64_t {100});
}

TEST(PenSmoothingBufferTest, BoundsBacklogLatency) {
  release_recorder_t recorder;
  {
    pen_smoothing_buffer_t buffer {recorder.callback()};
    for (auto i = 0; i < 20; ++i) {
      buffer.push(touch_port_t {}, make_pen(static_cast<float>(i), 0.5F));
    }
    // The backlog valve must drain the whole burst quickly even though
    // paced release alone would take noticeably longer.
    ASSERT_GE(recorder.wait_for(20, std::chrono::milliseconds(1000)), std::size_t {20});
  }

  EXPECT_EQ(recorder.snapshot().size(), std::size_t {20});
}

TEST(PenSmoothingBufferTest, FlushReleasesPendingSamples) {
  release_recorder_t recorder;
  pen_smoothing_buffer_t buffer {recorder.callback()};
  for (auto i = 0; i < 3; ++i) {
    buffer.push(touch_port_t {}, make_pen(static_cast<float>(i), 0.5F));
  }
  buffer.flush();

  ASSERT_GE(recorder.wait_for(3, std::chrono::milliseconds(500)), std::size_t {3});
  EXPECT_EQ(recorder.snapshot().size(), std::size_t {3});
}

TEST(PenSmoothingBufferTest, FlushesOnDestruction) {
  release_recorder_t recorder;
  {
    pen_smoothing_buffer_t buffer {recorder.callback()};
    for (auto i = 0; i < 3; ++i) {
      buffer.push(touch_port_t {}, make_pen(static_cast<float>(i), 0.5F));
    }
    // Deliberately no explicit flush: destruction must release the backlog
    // so terminal pen state such as a tip-up is never lost.
  }

  EXPECT_EQ(recorder.snapshot().size(), std::size_t {3});
}

TEST(PenSmoothingBufferTest, IdlesQuietlyWithoutInput) {
  release_recorder_t recorder;
  {
    pen_smoothing_buffer_t buffer {recorder.callback()};
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_EQ(recorder.snapshot().size(), std::size_t {0});
  }
  EXPECT_EQ(recorder.snapshot().size(), std::size_t {0});
}

/**
 * @brief Verify that samples lost between client timestamps are reconstructed.
 *
 * Three MOVE samples arrive with a four-interval gap in their session-clock
 * stamps; the buffer must release the three pushed samples plus three
 * synthesized ones, with the reconstruction falling between the known
 * neighbors in order, on evenly spaced client-clock stamps.
 */
TEST(PenSmoothingBufferTest, ReconstructsSamplesLostBetweenTimestamps) {
  release_recorder_t recorder;
  {
    pen_smoothing_buffer_t buffer {recorder.callback()};

    const auto stamped = [](std::uint32_t ts, float x) {
      pen_input_t pen = make_pen(x, 0.0F);
      pen.eventType = 0x03; // LI_TOUCH_EVENT_MOVE
      pen.clientTimestamp = ts;
      return pen;
    };

    // Seed the cadence with two stamped intervals, then lose three
    // samples. Start the clock at 1500 us: a zero stamp is the protocol's
    // 'no timestamp' sentinel, not a real sample time, and history must
    // not skip the first sample.
    buffer.push(touch_port_t {}, stamped(1500, 0.10F));
    buffer.push(touch_port_t {}, stamped(3000, 0.20F));
    buffer.push(touch_port_t {}, stamped(9000, 0.40F));

    ASSERT_GE(recorder.wait_for(6, std::chrono::milliseconds(2000)), std::size_t {6});
  }

  const auto samples = recorder.snapshot();
  ASSERT_EQ(samples.size(), std::size_t {6});

  // The three reconstructed samples sit between the known neighbors, in
  // order, on evenly spaced client-clock stamps.
  const auto &s2 = samples[2].pen;
  const auto &s3 = samples[3].pen;
  const auto &s4 = samples[4].pen;
  EXPECT_EQ(s2.clientTimestamp, std::uint32_t {4500});
  EXPECT_EQ(s3.clientTimestamp, std::uint32_t {6000});
  EXPECT_EQ(s4.clientTimestamp, std::uint32_t {7500});
  EXPECT_GT(s2.x, 0.20F);
  EXPECT_GT(s3.x, s2.x);
  EXPECT_GT(s4.x, s3.x);
  EXPECT_LT(s4.x, 0.40F);
}

/**
 * @brief Verify that clock discontinuities skip synthesis and reset the baseline.
 *
 * A wrap-spanning backwards stamp jump (the send-vs-acquisition offset at
 * hover/contact transitions) must not fabricate samples, and the pair
 * after the jump must re-seed the cadence rather than read as a loss gap.
 */
TEST(PenSmoothingBufferTest, SkipsSynthesisAcrossClockDiscontinuities) {
  release_recorder_t recorder;
  {
    pen_smoothing_buffer_t buffer {recorder.callback()};

    const auto stamped = [](std::uint32_t ts, float x) {
      pen_input_t pen = make_pen(x, 0.0F);
      pen.eventType = 0x03; // LI_TOUCH_EVENT_MOVE
      pen.clientTimestamp = ts;
      return pen;
    };

    // Seed the cadence with one stamped interval.
    buffer.push(touch_port_t {}, stamped(1500, 0.10F));
    buffer.push(touch_port_t {}, stamped(3000, 0.20F));

    // Stamps jump backwards by ~25 ms: the wrap-aware delta reads ~2^32 us.
    buffer.push(touch_port_t {}, stamped(3000U - 25000U, 0.30F));

    // The recovery pair re-seeds the baseline; nothing reconstructs.
    buffer.push(touch_port_t {}, stamped(3000U - 25000U + 1500U, 0.40F));

    ASSERT_GE(recorder.wait_for(4, std::chrono::milliseconds(2000)), std::size_t {4});
  }

  const auto samples = recorder.snapshot();
  ASSERT_EQ(samples.size(), std::size_t {4});
}

/**
 * @brief Verify that long forward stalls within the bound reconstruct, capped.
 *
 * A 400 ms stall between positional samples is genuine loss: measured live
 * during sustained drawing, 116-265 ms stalls produced the residual
 * straight segments the 100 ms bound silently refused. The buffer now
 * reconstructs forward stalls up to the bound, with the per-gap cap
 * bounding fabrication.
 */
TEST(PenSmoothingBufferTest, ReconstructsLongForwardStalls) {
  release_recorder_t recorder;
  {
    pen_smoothing_buffer_t buffer {recorder.callback()};

    const auto stamped = [](std::uint32_t ts, float x) {
      pen_input_t pen = make_pen(x, 0.0F);
      pen.eventType = 0x03; // LI_TOUCH_EVENT_MOVE
      pen.clientTimestamp = ts;
      return pen;
    };

    // Seed the cadence, then a 400 ms stall mid-stroke.
    buffer.push(touch_port_t {}, stamped(1500, 0.10F));
    buffer.push(touch_port_t {}, stamped(3000, 0.20F));
    buffer.push(touch_port_t {}, stamped(403000, 0.30F));

    // 3 pushed + 16 cap-bounded synthesized samples.
    ASSERT_GE(recorder.wait_for(19, std::chrono::milliseconds(3000)), std::size_t {19});
  }

  const auto samples = recorder.snapshot();
  ASSERT_EQ(samples.size(), std::size_t {19});

  // The 16 fills sit between the stall's endpoints, increasing in stamp.
  const auto &first_fill = samples[2].pen;
  const auto &last_fill = samples[17].pen;
  EXPECT_GT(first_fill.clientTimestamp, std::uint32_t {3000});
  EXPECT_LT(last_fill.clientTimestamp, std::uint32_t {403000});
  EXPECT_LT(first_fill.clientTimestamp, last_fill.clientTimestamp);
}

/**
 * @brief Verify that stalls beyond the bound refuse and reset the baseline.
 *
 * A forward stall beyond the reconstruction bound passes through
 * untouched, logs the skip, and resets the gap baseline; the recovery
 * pair re-seeds without synthesizing.
 */
TEST(PenSmoothingBufferTest, RefusesStallsBeyondTheReconstructionBound) {
  release_recorder_t recorder;
  {
    pen_smoothing_buffer_t buffer {recorder.callback()};

    const auto stamped = [](std::uint32_t ts, float x) {
      pen_input_t pen = make_pen(x, 0.0F);
      pen.eventType = 0x03; // LI_TOUCH_EVENT_MOVE
      pen.clientTimestamp = ts;
      return pen;
    };

    // Seed the cadence, then a 600 ms stall beyond the bound.
    buffer.push(touch_port_t {}, stamped(1500, 0.10F));
    buffer.push(touch_port_t {}, stamped(3000, 0.20F));
    buffer.push(touch_port_t {}, stamped(603000, 0.30F));

    // The recovery pair re-seeds the baseline without synthesizing.
    buffer.push(touch_port_t {}, stamped(604500, 0.40F));

    ASSERT_GE(recorder.wait_for(4, std::chrono::milliseconds(2000)), std::size_t {4});
  }

  const auto samples = recorder.snapshot();
  ASSERT_EQ(samples.size(), std::size_t {4});
}
