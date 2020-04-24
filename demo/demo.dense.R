path.prefix <- "."

source(paste0(path.prefix, "/R/mclogit_fast.R"))
source(paste0(path.prefix, "/R/r2.pseudo.R"))
Rcpp::sourceCpp(paste0(path.prefix, "/src/mclogit_fast.cpp"))

library(data.table)

if(FALSE) {
  # generate data
  set.seed(12345)
  BETA_A <- runif(1) * -0.1
  BETA_B <- runif(1) * 2
  BETA_C <- runif(1) * 0.5

  alts <- 1e5
  cases.outer <- alts / 10
  dat <- data.table(CASE = rep_len(1:cases.outer, alts),
                     A = runif(alts, 0, 500),
                     B = floor(runif(alts, -.5, 2)),
                     C = runif(alts, 10, 10000)/ 100)
  setorder(dat, CASE)

  # hide some preferences
  dat[, ETA := (BETA_A * A + BETA_B * B + BETA_C * C) * (1 + runif(.N, min=-0.05, max=0.05))]
  dat[, EXP_ETA := exp(ETA)]
  dat[, P := EXP_ETA / sum(exp(ETA)), by = CASE]
  dat[, Y := as.integer(P == max(P, na.rm=TRUE)), by = CASE]
  stopifnot(sum(dat$Y) == uniqueN(dat$CASE))

  # baseline implementation
  mc <- mclogit::mclogit(cbind(Y,CASE)~A+B+C,data=dat)
  summary(mc)
  
  # efficient implementation
  mf <- mclogit_fast(cbind(Y,CASE)~A+B+C,data=dat)
  summary(mf)

  # check that the results are equal
  stopifnot(all(abs(coef(mf) - coef(mc)) < 1e10))
  stopifnot(all(abs(vcov(mf) - vcov(mc)) < 1e10))
  stopifnot(abs(mf$deviance - mc$deviance) < 1e-3)
  stopifnot(abs(mf$null.deviance - mc$null.deviance) < 1e-3)
}
