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
