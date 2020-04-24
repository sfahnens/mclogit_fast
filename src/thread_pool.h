#pragma once

#include <condition_variable>
#include <atomic>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "./util.h"

namespace mclogit_fast {

struct thread_pool {
  explicit thread_pool(
      std::function<void()> initializer = [] {},
      std::function<void()> finalizer = [] {})
      : initializer_{std::move(initializer)},
        finalizer_{std::move(finalizer)},
        threads_active_{0},
        stop_{false},
        job_count_{0},
        counter_{0} {
    for (auto i = 0U; i < std::thread::hardware_concurrency(); ++i) {
      threads_.emplace_back([&, this] {
        initializer_();
        auto const finally = make_finally([&] { finalizer_(); });
        try {
          while (true) {
            {
              std::unique_lock<std::mutex> lk(mutex_);
              if (stop_ == true) {
                break;
              }
              cv_.wait(lk,
                       [&] { return stop_ == true || counter_ < job_count_; });
              if (stop_ == true) {
                break;
              }

              ++threads_active_;
            }

            if (pre_fn_) {
              pre_fn_();
            }

            while (true) {
              constexpr size_t kBatchSize = 1024;
              auto const begin = counter_.fetch_add(kBatchSize);
              for (auto idx = begin; idx < begin + kBatchSize; ++idx) {
                if (idx >= job_count_) {
                  if (post_fn_) {
                    post_fn_();
                  }
                  goto work_finished;
                }
                work_fn_(idx);
              }
            }

          work_finished:;
            {
              std::unique_lock<std::mutex> lk(mutex_);
              --threads_active_;
              cv_.notify_all();
            }
          }
        } catch (...) {
          std::unique_lock<std::mutex> lk(mutex_);
          if (!ex_ptr_) {  // keep only first exception
            ex_ptr_ = std::current_exception();
          }
          counter_ = job_count_;  // terminate early
          --threads_active_;
          cv_.notify_all();
        }
      });
    }
  }

  ~thread_pool() {
    {
      std::unique_lock<std::mutex> lk(mutex_);
      stop_ = true;
    }
    cv_.notify_all();
    std::for_each(begin(threads_), end(threads_), [](auto& t) { t.join(); });
  }

  thread_pool(thread_pool const&) noexcept = delete;             // NOLINT
  thread_pool& operator=(thread_pool const&) noexcept = delete;  // NOLINT
  thread_pool(thread_pool&&) noexcept = delete;                  // NOLINT
  thread_pool& operator=(thread_pool&&) noexcept = delete;       // NOLINT

  void execute(size_t const job_count, std::function<void(size_t)>&& work_fn,
               std::function<void()> pre_fn = nullptr,
               std::function<void()> post_fn = nullptr) {
    if (job_count == 0) {
      return;
    }

    verify(!ex_ptr_, "thread_pool: still have exception from previous run");

    work_fn_ = work_fn;
    pre_fn_ = pre_fn;
    post_fn_ = post_fn;

    {
      std::unique_lock<std::mutex> lk(mutex_);
      counter_ = 0;
      job_count_ = job_count;
    }

    cv_.notify_all();

    std::unique_lock<std::mutex> lk(mutex_);
    cv_.wait(lk,
             [&] { return threads_active_ == 0 && counter_ >= job_count_; });

    if (ex_ptr_) {
      std::rethrow_exception(ex_ptr_);
    }
  }

private:
  std::function<void()> initializer_, finalizer_;

  std::vector<std::thread> threads_;
  std::atomic_size_t threads_active_;
  bool stop_;
  std::exception_ptr ex_ptr_;

  std::mutex mutex_;
  std::condition_variable cv_;

  size_t job_count_;
  std::atomic_size_t counter_;
  std::function<void(size_t)> work_fn_;
  std::function<void()> pre_fn_, post_fn_;
};

}  // namespace mclogit_fast
