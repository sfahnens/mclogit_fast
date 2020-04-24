#pragma once

#include "util.h"

namespace mclogit_fast {

struct mclogit_fast_dense {
  mclogit_fast_dense(arma::colvec y, arma::colvec s, arma::colvec w,
                     arma::mat x)
      : y_{std::move(y)}, s_{std::move(s)}, w_{std::move(w)}, x_{std::move(x)} {
    verify_sorted(s_);
    verify(s_.n_rows > 0, "empty s not allowed");
    verify(x_.n_rows > 0 && x_.n_cols > 0, "empty x not allowed");
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

    std::cout << "dropping " << dropped_cols_.size() << " cols." << std::endl;
    if (!dropped_cols_.empty()) {
      x_.shed_cols(arma::uvec{dropped_cols_});
    }
  }

  void init_irls() {
    p_ = arma::colvec(y_.n_rows);
    yp_star_ = arma::colvec(y_.n_rows);
    xp_ = arma::mat(arma::size(x_));
    wp_ = arma::colvec(w_.n_rows);
    xpt_wp_xp_ = arma::mat(x_.n_cols, x_.n_cols);

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

  bool run_em(arma::colvec const s_outer) {
    verify_sorted(s_outer);

    if (!run_irls()) {
      return false;
    }
    for (em_iter_ = 0; em_iter_ < 100; ++em_iter_) {
      std::cout << "outer iteration " << em_iter_ << std::endl;
      update_weights_outer(s_outer);

      arma::colvec coef_old = coef_;
      if (!run_irls()) {
        return false;
      }

      double err = arma::sum(arma::square(coef_old - coef_));
      if (err < 1e-6) {
        std::cout << "outer converged" << std::endl;
        return true;
      }
    }

    return false;
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
    yp_star_ = eta_ + ((y_ - p_) / p_);
    by_grp(yp_star_, [&](auto lb, auto ub) {
      double acc = 0.;
      for (auto it = lb; it != ub; ++it) {
        acc += *it * proj_elem(yp_star_, it, p_);
      }
      std::for_each(lb, ub, [&](auto& e) { e -= acc; });
    });
    verify(yp_star_.is_finite(), "some yp is not finite");

    // unoptimized implementation:
    // arma::mat wp_diag = arma::diagmat(wp);
    // coef = arma::inv(xp.t() * ww_diag * xp) * xp.t() * ww_diag * yp_star_;

    // postcondition: xpt_wp_xp_, xp_ initialized
    compute_crossprod_xpt_wp_xp();
    xp_.each_col([&](arma::colvec& row) { row %= wp_; });  // elementwise prod
    coef_ = arma::solve(xpt_wp_xp_, xp_.t() * yp_star_);
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

    by_grp(y_, [&](auto lb, auto ub) {
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
    xp_ = x_;
    xp_.each_col([&](arma::vec& col) {
      by_grp(col, [&](auto lb, auto ub) {
        double acc = 0.;
        for (auto it = lb; it != ub; ++it) {
          acc += *it * proj_elem(col, it, p_);
        }
        std::for_each(lb, ub, [&](auto& e) { e -= acc; });
      });
    });

    wp_ = w_ % p_;
    if (!arma::all(wp_ > 0)) {
      std::cout << "some wp_ equal zero" << std::endl;
    }

    crossprod_weighted(xpt_wp_xp_, xp_, wp_);
  }

  void predict() {
    verify(arma::size(p_) == arma::size(eta_), "eta_to_p: size mismatch");

    p_ = arma::exp(eta_);
    by_grp(p_, [&](auto lb, auto ub) {
      double const sum = std::accumulate(lb, ub, 0.);
      std::for_each(lb, ub, [&](auto& e) { e /= sum; });
    });
  };

  // precondition predict() called and p_ up to date
  void update_weights_outer(arma::colvec const& s_outer) {
    equal_ranges_linear(s_outer, [&](auto lb, auto ub) {
      double denom = 0;
      for (auto it = lb; it != ub; ++it) {
        if (proj_elem(s_outer, it, y_) == 1.) {
          denom += proj_elem(s_outer, it, w_) * proj_elem(s_outer, it, p_);
        }
      }

      equal_ranges_linear(
          proj_it(s_outer, lb, s_), proj_it(s_outer, ub, s_),
          [&](auto lb_s, auto ub_s) {
            auto lb_y = proj_it(s_, lb_s, y_);
            auto ub_y = proj_it(s_, ub_s, y_);

            auto it_y = std::find_if(lb_y, ub_y, [](auto y) { return y == 1; });
            if (it_y == ub_y) {
              verify(lb_y != ub_y, "lb_y == ub_y, ???");
              std::cout << "max element: " << *std::max_element(lb_y, ub_y)
                        << std::endl;
            }

            verify(it_y != ub_y, "have no y==1");
            double const num =
                proj_elem(y_, it_y, w_) * proj_elem(y_, it_y, p_);

            auto lb_w = proj_it(s_, lb_s, w_);
            auto ub_w = proj_it(s_, ub_s, w_);
            for (auto it_w = lb_w; it_w != ub_w; ++it_w) {
              *it_w = num / denom;
            }
          });
    });
  }

  template <typename Vec, typename Fn>
  void by_grp(Vec& vec, Fn&& fn) {
    equal_ranges_linear(s_, [&](auto lb, auto ub) {
      auto lb_vec = proj_it(s_, lb, vec);
      auto ub_vec = proj_it(s_, ub, vec);
      fn(lb_vec, ub_vec);
    });
  }

  static void crossprod_weighted(arma::mat& out, arma::mat const& x,
                                 arma::colvec const& w) {
    out.zeros(x.n_cols, x.n_cols);

    for (auto i = 0ul; i < x.n_cols; ++i) {
      for (auto j = 0ul; j <= i; ++j) {
        double acc = 0;
        for (auto k = 0ul; k < x.n_rows; ++k) {
          acc += w(k) * x(k, i) * x(k, j);
        }
        out(i, j) = acc;
      }
    }

    for (auto i = 0ul; i < x.n_cols; ++i) {
      for (auto j = i + 1; j < x.n_cols; ++j) {
        out(i, j) = out(j, i);
      }
    }
  }

  arma::colvec y_;
  arma::colvec s_;
  arma::colvec w_;
  arma::mat x_;

  arma::colvec eta_;
  arma::colvec p_;

  arma::colvec yp_star_;
  arma::mat xp_;
  arma::colvec wp_;
  arma::mat xpt_wp_xp_;

  int irls_iter_, em_iter_;
  arma::colvec coef_;
  std::vector<arma::uword> dropped_cols_;
};

}  // namespace mclogit_fast
