// [[Rcpp::plugins(cpp14)]]
// [[Rcpp::depends(RcppArmadillo)]]

#define ARMA_USE_CXX11 1

// #define ARMA_EXTRA_DEBUG 1
// #define ARMA_CERR_STREAM std::cerr

#include "RcppArmadillo.h"

#include "./sparse_matrix.h"
#include "./util.h"

// [[Rcpp::export]]
SEXP matrix_cols(SEXP mat_x, SEXP mat_y) {
  mclogit_fast::sparse_matrix x{mat_x};
  mclogit_fast::sparse_matrix y{mat_y};
  verify(x.n_cols == y.n_cols, "BROKEN COLS");

  std::vector<int> col_ptrs, row_indices;
  col_ptrs.reserve(x.col_ptrs.size());
  row_indices.reserve(x.row_indices.size());
  std::vector<double> values;
  values.reserve(x.values.size());

  auto const* x_base = &x.row_indices[0];
  auto const* y_base = &y.row_indices[0];

  for (auto i = 0; i < x.n_cols; ++i) {
    col_ptrs.push_back(row_indices.size());

    auto const* x_begin = &x.row_indices[x.col_ptrs[i]];
    auto const* x_end = &x.row_indices[x.col_ptrs[i + 1]];
    auto const* y_begin = &y.row_indices[y.col_ptrs[i]];
    auto const* y_end = &y.row_indices[y.col_ptrs[i + 1]];

    for (auto x_it = x_begin; x_it != x_end; ++x_it) {
      for (auto y_it = y_begin; y_it != y_end; ++y_it) {
        row_indices.push_back(*x_it + *y_it * x.n_rows);
        values.push_back(x.values[std::distance(x_base, x_it)] *
                         y.values[std::distance(y_base, y_it)]);
      }
    }
  }
  col_ptrs.push_back(row_indices.size());

  return mclogit_fast::finish_matrix(
      Rcpp::IntegerVector::create(x.n_rows * y.n_rows, x.n_cols),
      Rcpp::IntegerVector{begin(col_ptrs), end(col_ptrs)},
      Rcpp::IntegerVector{begin(row_indices), end(row_indices)},
      Rcpp::NumericVector{begin(values), end(values)});
}

// [[Rcpp::export]]
SEXP fac_to_sparse(Rcpp::IntegerVector vec, int levels) {
  verify(vec.size() > 0 && levels > 0, "fac_to_sparse: invalid dimensions");
  std::vector<std::vector<int>> cols(levels, std::vector<int>{});

  std::vector<int> col_ptrs, row_indices;
  col_ptrs.reserve(vec.size());
  row_indices.reserve(vec.size());

  for (auto i = 0; i < vec.size(); ++i) {
    auto const v = vec[i];
    col_ptrs.push_back(row_indices.size());

    if (v != NA_INTEGER) {
      row_indices.push_back(v - 1);
    }
  }
  col_ptrs.push_back(row_indices.size());

  return mclogit_fast::finish_matrix(
      Rcpp::IntegerVector::create(levels, row_indices.size()),
      Rcpp::IntegerVector{begin(col_ptrs), end(col_ptrs)},
      Rcpp::IntegerVector{begin(row_indices), end(row_indices)},
      Rcpp::NumericVector(row_indices.size(), 1.));
}

// [[Rcpp::export]]
Rcpp::NumericVector wsum_per_col_sparse(SEXP x_exp, Rcpp::NumericVector y,
                                        Rcpp::NumericVector w) {
  mclogit_fast::sparse_matrix x{x_exp};
  verify(x.n_rows == y.size() && y.size() == w.size(),
         "nnz_per_col_dense: size mismatch");

  std::vector<double> vec;
  for (auto i = 0; i < x.n_cols; ++i) {
    double acc = 0.;
    for (auto j = x.col_ptrs[i]; j < x.col_ptrs[i + 1]; ++j) {
      auto const row_idx = x.row_indices[j];
      acc += y[row_idx] * w[row_idx];
    }
    vec.push_back(acc);
  }

  Rcpp::NumericVector result{begin(vec), end(vec)};

  auto x_obj = Rcpp::as<Rcpp::RObject>(x_exp);
  if (x_obj.hasAttribute("Dimnames") &&
      Rcpp::as<Rcpp::List>(x_obj.attr("Dimnames")).size() == 2) {
    result.attr("names") = Rcpp::as<Rcpp::List>(x_obj.attr("Dimnames"))[1];
  }
  return result;
}
