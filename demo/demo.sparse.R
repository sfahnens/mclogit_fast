path.prefix <- "."

source(paste0(path.prefix, "/R/mclogit_fast.R"))
source(paste0(path.prefix, "/R/r2.pseudo.R"))
source(paste0(path.prefix, "/R/my.sparse.model.matrix.R"))

Rcpp::sourceCpp(paste0(path.prefix, "/src/mclogit_fast.cpp"))
Rcpp::sourceCpp(paste0(path.prefix, "/src/matrix_tools.cpp"))

library(data.table)

if(FALSE) {
  # generate data
  set.seed(12345)
  BETA_A <- runif(1) * -0.1
  BETA_B <- runif(1) * 2
  BETA_C <- runif(1) * 0.5

  shift <- 0.05

  alts <- 1e6
  cases.outer <- alts / 10
  dat <- data.table(CASE = rep_len(1:cases.outer, alts),
                     A = as.logical(runif(alts) > .8),
                     B = as.factor(as.integer(runif(alts) * 3)),
                     C = runif(alts, 0, 240),
                     D = floor(runif(alts, -.49, 2)),
                     E = runif(alts, 10, 10000)/ 100)
  setorder(dat, CASE)

  # hide some preferences
  dat[, ETA := (BETA_A * ( 1 + shift * A + shift * as.integer(B)) * C +
                 BETA_B * ( 1 + shift * A + shift * as.integer(B)) * D +
                 BETA_C * ( 1 + shift * A + shift * as.integer(B)) * E) *
         (1 + runif(.N, min=-0.05, max=0.05))]
  dat[, EXP_ETA := exp(ETA)]
  dat[, P := EXP_ETA / sum(exp(ETA)), by = CASE]
  dat[, Y := as.integer(P == max(P, na.rm=TRUE)), by = CASE]
  stopifnot(sum(dat$Y) == uniqueN(dat$CASE))

  # baseline implementation
  mc <- mclogit::mclogit(cbind(Y,CASE)~B:A:(C+D+E),data=dat)
  summary(mc)

  # efficient parallel, sparse implementation
  mf <- mclogit_fast(cbind(Y,CASE)~ B:A:(C+D+E),data=dat, sparse = TRUE)
  mf <- strip.mclogit_fast(mf)
  summary(mf)

  # check that the results are equal
  stopifnot(all(abs(coef(m0) - coef(mf)) < 1e10))
  stopifnot(all(abs(vcov(m0) - vcov(mf)) < 1e10))
  stopifnot(abs(m0$deviance - mf$deviance) < 1e-3)
  stopifnot(abs(m0$null.deviance - mf$null.deviance) < 1e-3)
}
