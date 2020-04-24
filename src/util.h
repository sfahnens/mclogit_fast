#pragma once

// #############################################################################
//   from rutl/verify.h
// #############################################################################
#ifndef log_err
#define log_err(M, ...) fprintf(stderr, "[ERR] " M "\n", ##__VA_ARGS__);
#endif

#ifdef verify
#undef verify
#endif

#define verify(A, M, ...)                         \
  if (!(A)) {                                     \
    log_err(M, ##__VA_ARGS__);                    \
    throw Rcpp::exception(M, __FILE__, __LINE__); \
  }

#define verify_sorted(V) \
  verify(std::is_sorted(std::begin(V), std::end(V)), #V " not sorted!");

namespace mclogit_fast {

// #############################################################################
//   from rutl/equal_ranges_linear.h
// #############################################################################
template <typename Iterator, typename F>
void equal_ranges_linear(Iterator begin, Iterator end, F&& func) {
  auto lower = begin;
  while (lower != end) {
    auto upper = lower;
    while (upper != end && *lower == *upper) {
      ++upper;
    }
    func(lower, upper);
    lower = upper;
  }
}

template <typename Container, typename F>
void equal_ranges_linear(Container&& c, F&& func) {
  equal_ranges_linear(std::begin(c), std::end(c), std::forward<F&&>(func));
}

template <typename Iterator, typename Eq, typename F>
void equal_ranges_linear(Iterator begin, Iterator end, Eq&& eq, F&& func) {
  auto lower = begin;
  while (lower != end) {
    auto upper = lower;
    while (upper != end && eq(*lower, *upper)) {
      ++upper;
    }
    func(lower, upper);
    lower = upper;
  }
}

template <typename Container, typename Eq, typename F>
void equal_ranges_linear(Container&& c, Eq&& eq, F&& func) {
  equal_ranges_linear(std::begin(c), std::end(c), std::forward<Eq&&>(eq),
                      std::forward<F&&>(func));
}

// #############################################################################
//   from rutl/finally.h
// #############################################################################
template <typename DestructFun>
struct finally {
  explicit finally(DestructFun&& destruct)
      : destruct_(std::forward<DestructFun>(destruct)) {}

  finally(finally const&) = delete;
  finally& operator=(finally const&) = delete;

  finally(finally&& o) noexcept : destruct_{std::move(o.destruct_)} {
    o.exec_ = false;
  }

  finally& operator=(finally&& o) noexcept {
    destruct_ = std::move(o.destruct_);
    o.exec_ = false;
    return *this;
  }

  ~finally() {
    if (exec_) {
      destruct_();
    }
  }
  bool exec_{true};
  DestructFun destruct_;
};

template <typename DestructFun>
finally<DestructFun> make_finally(DestructFun&& destruct) {
  return finally<DestructFun>(std::forward<DestructFun>(destruct));
}

// #############################################################################
//   from rutl/proj_cols.h
// #############################################################################
template <typename BaseVec, typename BaseIt>
std::size_t proj_idx(BaseVec const& base, BaseIt const it) {
  typename BaseVec::const_iterator const_begin = std::begin(base);
  typename BaseVec::const_iterator const_it = it;

  return std::distance(const_begin, const_it);
}

template <typename BaseVec, typename BaseIt, typename OtherVec>
auto proj_it(BaseVec const& base, BaseIt const it, OtherVec& other)
    -> decltype(std::begin(other)) {
  return std::begin(other) + proj_idx(base, it);
}

template <typename BaseVec, typename BaseIt, typename OtherVec>
auto proj_elem(BaseVec const& base, BaseIt const it, OtherVec& other)
    -> decltype(other.at(0)) {
  return other.at(proj_idx(base, it));
}

}  // namespace mclogit_fast
