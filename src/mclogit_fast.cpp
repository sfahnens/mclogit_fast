// [[Rcpp::plugins(cpp14)]]
// [[Rcpp::depends(RcppArmadillo)]]

#define ARMA_USE_CXX11 1

// #define ARMA_EXTRA_DEBUG 1
// #define ARMA_CERR_STREAM std::cerr
// #define ARMA_DONT_USE_OPENMP 1

#include "RcppArmadillo.h"

#include "./mclogit_fast_dense_fit.h"
#include "./mclogit_fast_dense_fit_pll.h"
#include "./mclogit_fast_sparse_fit_pll.h"

#include "./util.h"

// [[Rcpp::export]]
Rcpp::List mclogit_fast_dense_fit(Rcpp::NumericVector y, Rcpp::NumericVector s,
                                  Rcpp::NumericVector w,
                                  Rcpp::NumericMatrix x) {
  auto flush_cout = mclogit_fast::make_finally([] {
    std::cout << std::flush;
    std::cerr << std::flush;
  });

  mclogit_fast::mclogit_fast_dense logit{
      arma::colvec(y.begin(), y.size(), false, true),
      arma::colvec(s.begin(), s.size(), false, true),
      arma::colvec(w.begin(), w.size(), false, true),
      arma::mat(x.begin(), x.nrow(), x.ncol(), false, true)};

  logit.prepare();
  logit.init_irls();

  if (logit.run_irls()) {
    return Rcpp::List::create(
        Rcpp::Named("coef", Rcpp::NumericVector{std::begin(logit.coef_),
                                                std::end(logit.coef_)}),
        Rcpp::Named("covmat", logit.get_covmat()),
        Rcpp::Named("converged", true),             //
        Rcpp::Named("iter", logit.irls_iter_),      //
        Rcpp::Named("deviance", logit.deviance()),  //
        Rcpp::Named("null.deviance", logit.null_deviance()));
  }

  return Rcpp::List::create();
}

// [[Rcpp::export]]
Rcpp::List mclogit_fast_dense_fit_pll(Rcpp::NumericVector y,
                                      Rcpp::NumericVector s,
                                      Rcpp::NumericVector w,
                                      Rcpp::NumericMatrix x) {
  auto flush_cout = mclogit_fast::make_finally([] {
    std::cout << std::flush;
    std::cerr << std::flush;
  });

  mclogit_fast::mclogit_fast_dense_pll logit{
      arma::colvec(y.begin(), y.size(), false, true),
      arma::colvec(s.begin(), s.size(), false, true),
      arma::colvec(w.begin(), w.size(), false, true),
      arma::mat(x.begin(), x.nrow(), x.ncol(), false, true)};

  logit.prepare();
  logit.init_irls();

  if (logit.run_irls()) {
    return Rcpp::List::create(
        Rcpp::Named("coef", Rcpp::NumericVector{std::begin(logit.coef_),
                                                std::end(logit.coef_)}),
        Rcpp::Named("covmat", logit.get_covmat()),
        Rcpp::Named("converged", true),             //
        Rcpp::Named("iter", logit.irls_iter_),      //
        Rcpp::Named("deviance", logit.deviance()),  //
        Rcpp::Named("null.deviance", logit.null_deviance()));
  }

  return Rcpp::List::create();
}

// [[Rcpp::export]]
Rcpp::List mclogit_fast_sparse_fit_pll(Rcpp::NumericVector y,
                                       Rcpp::NumericVector s,
                                       Rcpp::NumericVector w, SEXP x) {
  auto flush_cout = mclogit_fast::make_finally([] {
    std::cout << std::flush;
    std::cerr << std::flush;
  });

  mclogit_fast::mclogit_fast_sparse_pll logit{
      arma::colvec(y.begin(), y.size(), false, true),
      arma::colvec(s.begin(), s.size(), false, true),
      arma::colvec(w.begin(), w.size(), false, true),
      mclogit_fast::sparse_matrix{x}};

  logit.prepare();
  logit.init_irls();

  if (logit.run_irls()) {
    return Rcpp::List::create(
        Rcpp::Named("coef", Rcpp::NumericVector{std::begin(logit.coef_),
                                                std::end(logit.coef_)}),
        Rcpp::Named("covmat", logit.get_covmat()),
        Rcpp::Named("converged", true),                       //
        Rcpp::Named("iter", logit.irls_iter_),                //
        Rcpp::Named("deviance", logit.deviance()),            //
        Rcpp::Named("null.deviance", logit.null_deviance()),  //
        Rcpp::Named("ll", logit.log_lik()));
  }

  return Rcpp::List::create();
}
