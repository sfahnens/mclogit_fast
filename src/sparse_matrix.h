#pragma once

#include "./util.h"

namespace mclogit_fast {

struct sparse_matrix {
  explicit sparse_matrix(SEXP exp) : mat_{exp} {
    verify(Rcpp::as<std::string>(mat_.slot("class")) == "dgCMatrix",
           "matrix not a dgCMatrix");

    Rcpp::IntegerVector dims = mat_.slot("Dim");
    verify(dims.size() == 2, "invalid Dims");
    n_rows = dims[0];
    n_cols = dims[1];

    col_ptrs = mat_.slot("p");
    row_indices = mat_.slot("i");
    values = mat_.slot("x");

    verify(col_ptrs.size() == n_cols + 1, "sparse_matrix col ptr");
    verify(row_indices.size() == values.size(), "sparse_matrix entry mismatch");
  }

  Rcpp::S4 mat_;
  int n_rows, n_cols;
  Rcpp::IntegerVector col_ptrs;
  Rcpp::IntegerVector row_indices;
  Rcpp::DoubleVector values;
};

inline SEXP finish_matrix(Rcpp::IntegerVector dim,  //
                          Rcpp::IntegerVector col_ptrs,
                          Rcpp::IntegerVector row_indices,
                          Rcpp::NumericVector values) {
  Rcpp::S4 s(std::string{"dgCMatrix"});
  s.slot("i") = row_indices;
  s.slot("p") = col_ptrs;
  s.slot("x") = values;
  s.slot("Dim") = dim;
  return s;
}

}  // namespace mclogit_fast
