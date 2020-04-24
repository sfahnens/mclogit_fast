path.prefix <- "."

source(paste0(path.prefix, "/R/mclogit_fast.R"))
source(paste0(path.prefix, "/R/r2.pseudo.R"))
source(paste0(path.prefix, "/R/my.sparse.model.matrix.R"))

Rcpp::sourceCpp(paste0(path.prefix, "/src/mclogit_fast.cpp"))
Rcpp::sourceCpp(paste0(path.prefix, "/src/matrix_tools.cpp"))

library(data.table)
library(mclogit)

args <- commandArgs(trailingOnly = TRUE)
opt <- unserialize(charToRaw(gsub("\\\\n", "\n", args)))[[1]]

set.seed(12345)
BETA_M <- runif(1) * -0.1
BETA_N <- runif(1) * 2
BETA_O <- runif(1) * 0.5

shift <- 0.05

nrow <- opt$nrow
cases <- nrow / 10
dat <- data.table(CASE = rep_len(1:cases, nrow),
                  A = as.logical(runif(nrow) > .8),
                  B = as.factor(as.integer(runif(nrow) * 3)),
                  C = as.logical(runif(nrow) < .3),
                  D = as.factor(as.integer(runif(nrow) * 3)),
                  M = runif(nrow, 0, 240),
                  N = floor(runif(nrow, -.49, 2)),
                  O = runif(nrow, 10, 10000)/ 100)
setorder(dat, CASE)

dat[, ETA := (BETA_M * ( 1 + shift * A + shift * as.integer(B) - shift * C - shift * as.integer(D)) * M +
              BETA_N * ( 1 + shift * A + shift * as.integer(B) - shift * C - shift * as.integer(D)) * N +
              BETA_O * ( 1 + shift * A + shift * as.integer(B) - shift * C - shift * as.integer(D)) * O) *
      (1 + runif(.N, min=-0.05, max=0.05))]
dat[, P := exp(ETA) / sum(exp(ETA)), by = CASE]
dat[, Y := as.integer(P == max(P, na.rm=TRUE)), by = CASE]
stopifnot(sum(dat$Y) == uniqueN(dat$CASE))

dat[, ETA := NULL]
dat[, P := NULL]
print(gc(TRUE))

cat(sprintf("dataset size: %s\n", object.size(dat)))

print("RUN BENCHMARK")
print(opt$fn)
st <- system.time({
  m <- do.call(opt$fn, modifyList(list(formula=cbind(Y,CASE)~A:B:C:D:(M+N+O),data=dat), opt$args))
})
print(class(m))
print(coef(m))

cat(sprintf("system.time: user %s\n", st[1]))
cat(sprintf("system.time: sys %s\n", st[2]))
cat(sprintf("system.time: wall %s\n", st[3]))
