path.prefix <- "."

source(paste0(path.prefix, "/R/mclogit_fast.R"))
source(paste0(path.prefix, "/R/r2.pseudo.R"))
Rcpp::sourceCpp(paste0(path.prefix, "/src/mclogit_fast.cpp"))

library(data.table)
library(mclogit)

args <- commandArgs(trailingOnly = TRUE)
opt <- unserialize(charToRaw(gsub("\\\\n", "\n", args)))[[1]]
print(opt)

set.seed(12345)
BETA_A <- runif(1) * -0.1
BETA_B <- runif(1) * 2
BETA_C <- runif(1) * 0.5
BETA_D <- runif(1) * -2
BETA_E <- runif(1) * 0.1

nrow <- opt$nrow
cases <- nrow / 10
dat <- data.table(CASE = rep_len(1:cases, nrow),
                    A = runif(nrow, 0, 500),
                    B = floor(runif(nrow, -.5, 2)),
                    C = runif(nrow, 10, 10000)/ 100,
                    D = floor(runif(nrow, -.5, 2)),
                    E = runif(nrow, 0, 500))
setorder(dat, CASE)

dat[, ETA := (BETA_A * A + BETA_B * B + BETA_C * C + BETA_D * D + BETA_E * E) * (1 + runif(.N, min=-0.05, max=0.05))]
dat[, P :=  exp(ETA) / sum(exp(ETA)), by = CASE]
dat[, Y := as.integer(P == max(P, na.rm=TRUE)), by = CASE]
stopifnot(sum(dat$Y) == uniqueN(dat$CASE))

dat[, ETA := NULL]
dat[, P := NULL]
print(gc(TRUE))

# reset memory usage high water mark
system2("sh", c("-c", "echo", "5", sprintf("/proc/%s/clear_refs", Sys.getpid())))

cat(sprintf("dataset size: %s\n", object.size(dat)))

print("RUN BENCHMARK")
print(opt$fn)
st <- system.time({
  m <- do.call(opt$fn, modifyList(list(formula=cbind(Y,CASE)~A+B+C+D+E,data=dat), opt$args))
})
print(class(m))
print(coef(m))

cat(sprintf("system.time: user %s\n", st[1]))
cat(sprintf("system.time: sys %s\n", st[2]))
cat(sprintf("system.time: wall %s\n", st[3]))

cat(system2("cat", sprintf("/proc/%s/status", Sys.getpid())))
