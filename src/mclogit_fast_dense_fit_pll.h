#pragma once

#include "./thread_pool.h"
#include "./util.h"

namespace mclogit_fast {

thread_local static arma::colvec tmp_wp_;
thread_local static std::vector<double> tmp_yp_star_;
thread_local static arma::mat tmp_xp_;

thread_local static arma::mat tmp_xpt_wp_xp_;
thread_local static arma::colvec tmp_coef_alt_;

struct mclogit_fast_dense_pll {
  mclogit_fast_dense_pll(arma::colvec y, arma::colvec s, arma::colvec w,
                         arma::mat x)
      : y_{std::move(y)},
        s_{std::move(s)},
        w_{std::move(w)},
        x_{std::move(x)},
        tp_{[this] {},
            [&] {  // explicitly clear to prevent memory leaks
              tmp_wp_ = arma::colvec{};
              tmp_yp_star_ = std::vector<double>{};
              tmp_xp_ = arma::mat{};
              tmp_xpt_wp_xp_ = arma::mat{};
              tmp_coef_alt_ = arma::colvec{};
            }} {
    verify_sorted(s_);
    verify(s_.n_rows > 0, "empty s not allowed");
    verify(x_.n_rows > 0 && x_.n_cols > 0, "empty x not allowed");

    equal_ranges_linear(s_, [&](auto lb, auto ub) {
      auto const y_sum =
          std::accumulate(proj_it(s_, lb, y_), proj_it(s_, ub, y_), 0.);
      verify(y_sum > 0, "have empty y_sum");

      groups_.emplace_back(std::distance(std::begin(s_), lb));
    });
    groups_.emplace_back(s_.size());
  }

  void prepare() {
    adjust_weights();
    drop_const_coefs();
  }

  void adjust_weights() {
    by_grp(w_, [&](auto lb, auto ub) {
      double w_sum = 0.;
      for (auto it = lb; it != ub; ++it) {
        w_sum += *it * proj_elem(w_, it, y_);
      }
      for (auto it = lb; it != ub; ++it) {
        *it = w_sum;
      }
    });

    if (!std::all_of(std::begin(y_), std::end(y_),
                     [](auto y) { return y == 0. || y == 1.; })) {
      y_ /= w_;
    }

    for (arma::uword i = 0u; i < y_.n_rows; ++i) {
      if (w_(i) == 0) {
        y_(i) = 0;
      }
    }
  }

  void drop_const_coefs() {
    constexpr auto kInitial = -std::numeric_limits<double>::infinity();
    constexpr auto kExcluded = std::numeric_limits<double>::infinity();

    for (arma::uword i = 0u; i < x_.n_cols; ++i) {
      double curr_x = kInitial;
      double curr_s = kInitial;

      for (arma::uword j = 0; j < x_.n_rows; ++j) {
        if (s_[j] != curr_s) {
          curr_s = s_[j];
          curr_x = kInitial;
        }

        if (curr_x == kExcluded) {
          continue;  // "non-finite" variance
        }

        auto const x = x_(j, i);
        if (!std::isfinite(x)) {
          curr_x = kExcluded;
        } else if (curr_x == kInitial) {
          curr_x = x;
        } else if (curr_x != x) {
          goto dont_drop_col;
        }
      }
      dropped_cols_.push_back(i);
    dont_drop_col:;
    }

    std::cout << "drop " << dropped_cols_.size() << " cols. " << std::endl;
    if (!dropped_cols_.empty()) {
      x_.shed_cols(arma::uvec{dropped_cols_});
    }
  }

  void init_irls() {
    p_ = arma::colvec(y_.n_rows);

    xpt_wp_xp_ = arma::mat(x_.n_cols, x_.n_cols);
    coef_alt_ = arma::colvec(y_.n_cols);

    eta_ = (w_ % y_) + .5;  // elementwise product
    by_grp(eta_, [&](auto lb, auto ub) {
      double const sum = std::accumulate(lb, ub, 0.);
      std::for_each(lb, ub, [&](auto& e) { e = std::log(e / sum); });

      double const log_mean =
          std::accumulate(lb, ub, 0.) / std::distance(lb, ub);
      std::for_each(lb, ub, [&](auto& e) { e -= log_mean; });
    });
    predict();
  }

  bool run_irls() {
    arma::colvec last_coef = coef_;
    double last_dev = deviance();
    for (irls_iter_ = 0; irls_iter_ < 25; ++irls_iter_) {
      irls_step();

      double dev = deviance();
      if (!std::isfinite(dev)) {
        if (irls_iter_ == 0) {
          std::cout << "infinite deviance (first iter)\n";
          return false;
        }

        for (auto j = 0; j < 25; ++j) {
          if (std::isfinite(dev)) {
            break;
          }
          std::cout << "iter " << irls_iter_ << " shrink " << j << std::endl;

          coef_ = (coef_ + last_coef) / 2;
          eta_ = x_ * coef_;
          predict();

          dev = deviance();
          last_coef = coef_;
        }

        if (!std::isfinite(dev)) {
          std::cout << "infinite deviance\n";
          return false;
        }
      }

      if (std::fabs(dev - last_dev) / std::fabs(0.1 + dev) < 1e-8) {
        return true;
      } else {
        std::cout << "iter " << irls_iter_ << " deviance " << dev << std::endl;
        verify(irls_iter_ == 0 || dev < last_dev, "divergence!");
      }
      last_coef = coef_;
      last_dev = dev;
    }

    std::cout << "max iterations reached\n";
    return false;
  }

  void irls_step() {
    // arma::mat wp_diag = arma::diagmat(wp);
    // coef = arma::inv(xp.t() * ww_diag * xp) * xp.t() * ww_diag * yp_star_;

    // postcondition: xpt_wp_xp_, coef_alt_ initialized
    compute_crossprod_xpt_wp_xp();
    coef_ = arma::solve(xpt_wp_xp_, coef_alt_);
    eta_ = x_ * coef_;

    predict();
  }

  arma::mat get_covmat() {
    compute_crossprod_xpt_wp_xp();
    return arma::inv(xpt_wp_xp_);
  }

  double deviance() const {
    double dev = 0;
    for (auto i = 0ul; i < y_.n_rows; ++i) {
      if (y_.at(i) > 0) {
        dev +=
            2 * w_.at(i) * y_.at(i) * (std::log(y_.at(i)) - std::log(p_.at(i)));
      }
    }
    return dev;
  }

  double null_deviance() {
    double null_dev = 0;

    by_grp_serial(y_, [&](auto lb, auto ub) {
      double p0 = 1. / std::distance(lb, ub);

      for (auto it = lb; it != ub; ++it) {
        if (*it > 0) {
          null_dev += 2 * proj_elem(y_, it, w_) * (*it) *
                      (std::log(*it) - std::log(p0));
        }
      }
    });
    return null_dev;
  }

  void compute_crossprod_xpt_wp_xp() {
    std::mutex mtx;
    xpt_wp_xp_.zeros(x_.n_cols, x_.n_cols);
    coef_alt_.zeros(x_.n_cols);

    by_grp_pre_post(
        p_,
        [&] {
          tmp_xpt_wp_xp_.zeros(x_.n_cols, x_.n_cols);
          tmp_coef_alt_.zeros(x_.n_cols);
        },
        [&](auto p_lb, auto p_ub) {
          auto const offset = std::distance(std::begin(p_), p_lb);
          auto const size = std::distance(p_lb, p_ub);

          tmp_xp_.zeros(static_cast<arma::uword>(size),
                        static_cast<arma::uword>(x_.n_cols));
          for (arma::uword i = 0; i < x_.n_cols; ++i) {
            double acc = 0.;
            for (auto j = 0; j < size; ++j) {
              acc += x_(offset + j, i) * p_lb[j];
            }
            for (auto j = 0; j < size; ++j) {
              tmp_xp_(j, i) = x_(offset + j, i) - acc;
            }
          }

          tmp_wp_.resize(size);
          for (auto i = 0; i < size; ++i) {
            tmp_wp_[i] = w_[offset + i] * p_[offset + i];
            verify(tmp_wp_[i] > 0, "wp not equal zero");
          }

          tmp_xpt_wp_xp_ += tmp_xp_.t() * arma::diagmat(tmp_wp_) * tmp_xp_;

          double acc = 0.;
          tmp_yp_star_.clear();
          for (auto i = 0; i < size; ++i) {
            tmp_yp_star_.push_back(
                eta_[offset + i] +
                ((y_[offset + i] - p_[offset + i]) / p_[offset + i]));
            acc += tmp_yp_star_.back() * p_[offset + i];
          }
          for (auto i = 0; i < size; ++i) {
            tmp_yp_star_[i] -= acc;
            verify(std::isfinite(tmp_yp_star_[i]), "some yp is not finite");
          }

          for (auto i = 0ul; i < x_.n_cols; ++i) {
            double acc = 0.;
            for (auto j = 0; j < size; ++j) {
              acc += tmp_xp_(j, i) * tmp_wp_[j] * tmp_yp_star_[j];
            }
            tmp_coef_alt_(i) += acc;
          }
        },
        [&] {
          std::lock_guard<std::mutex> g{mtx};
          xpt_wp_xp_ += tmp_xpt_wp_xp_;
          coef_alt_ += tmp_coef_alt_;
        });
  }

  void predict() {
    verify(arma::size(p_) == arma::size(eta_), "eta_to_p: size mismatch");

    p_ = arma::exp(eta_);
    by_grp(p_, [&](auto lb, auto ub) {
      double const sum = std::accumulate(lb, ub, 0.);
      std::for_each(lb, ub, [&](auto& e) { e /= sum; });
    });
  };

  template <typename Vec, typename Fn>
  void by_grp(Vec& vec, Fn&& fn) {
    tp_.execute(groups_.size() - 1, [&](auto const i) {
      auto lb = &vec[groups_[i]];
      auto ub = &vec[groups_[i + 1]];
      fn(lb, ub);
    });
  }

  template <typename Vec, typename Fn>
  void by_grp_serial(Vec& vec, Fn&& fn) {
    equal_ranges_linear(s_, [&](auto lb, auto ub) {
      auto lb_vec = proj_it(s_, lb, vec);
      auto ub_vec = proj_it(s_, ub, vec);
      fn(lb_vec, ub_vec);
    });
  }

  template <typename Vec, typename PreFn, typename WorkFn, typename PostFn>
  void by_grp_pre_post(Vec& vec, PreFn&& pre_fn, WorkFn&& work_fn,
                       PostFn&& post_fn) {
    tp_.execute(
        groups_.size() - 1,
        [&](auto const i) {
          auto lb = &vec[groups_[i]];
          auto ub = &vec[groups_[i + 1]];
          work_fn(lb, ub);
        },
        pre_fn, post_fn);
  }

  arma::colvec y_;
  arma::colvec s_;
  arma::colvec w_;
  arma::mat x_;

  arma::colvec eta_;
  arma::colvec p_;

  std::vector<size_t> groups_;
  thread_pool tp_;

  arma::mat xpt_wp_xp_;
  arma::colvec coef_alt_;

  int irls_iter_, em_iter_;
  arma::colvec coef_;
  std::vector<arma::uword> dropped_cols_;
};

}  // namespace mclogit_fast
