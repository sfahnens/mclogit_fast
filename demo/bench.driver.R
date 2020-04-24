library(data.table)

run.bench <- function(which, opt) {
  out <- system2("/usr/bin/time",
                 c("-v", "Rscript",
                   sprintf("demo/bench.%s.R", which),
                   shQuote(gsub("\n", "\\\\n", (rawToChar(serialize(list(opt), NULL, ascii = TRUE)))))),
                 stdout = TRUE, stderr = TRUE)
  OUT <<- copy(out)
  return(out)
}

parse.out <- function(out, re) {
  rows <- regmatches(out,  regexec(re, out))
  return(as.numeric(Find(function(r) length(r) == 2, rows)[2]))
}

run.repeated <- function(n, which, confs) {
  rbindlist(lapply(confs, function(conf) {
    return(rbindlist(lapply(1:n, function(i) {
      cat(sprintf("run: %s %s(%s) - %s\n", which, conf$name, conf$opt$nrow, i))
      out <- run.bench(which, opt = conf$opt)
      perf <- data.table(
        NAME = conf$name,
        NROW = conf$opt$nrow,
        ITER = i,
        USER = parse.out(out, "^system.time: user (.*?)$"),
        WALL = parse.out(out, "^system.time: wall (.*?)$"),
        RSS_BYTE = parse.out(out, "^\tMaximum resident set size \\(kbytes\\): (.*?)$") * 1024,
        DATA_BYTE = parse.out(out, "^dataset size: (.*?)$")
      )
      return(perf)
    })))
  }))
}

printMarkdownTable <- function(dat) {
  cat(paste(colnames(dat), collapse = "|"))
  cat("\n")

  cat(paste(rep("---", ncol(dat)), collapse = "|"))
  cat("\n")

  dat.t <- transpose(dat)
  for(i in 1:ncol(dat.t)) {
    cat(paste(dat.t[[i]], collapse = "|"))
    cat("\n")
  }
}

if(FALSE) {
  # install.packages(c("data.table", "Rcpp", "RcppArmadillo", "mclogit"))
  repeats <- 3

  # --------------------------------------------------------------------------
  # dense benchmarks
  # --------------------------------------------------------------------------
  perf.dense.s <- run.repeated(1, "dense", list(
    list(
      name = "(1) mclogit",
      opt = list(fn="mclogit", nrow = 1e6, args=list())
    ),
    list(
      name = "(2) mclogit_fast(ser)",
      opt = list(fn="mclogit_fast", nrow = 1e6, args=list(sparse = FALSE, parallel=FALSE))
    ),
    list(
      name = "(3) mclogit_fast(pll)",
      opt = list(fn="mclogit_fast", nrow = 1e6, args=list(sparse = FALSE, parallel=TRUE))
    )
  ))
  perf.dense.m <- run.repeated(1, "dense", list(
    list(
      name = "(1) mclogit",
      opt = list(fn="mclogit", nrow = 1e7, args=list())
    ),
    list(
      name = "(2) mclogit_fast(ser)",
      opt = list(fn="mclogit_fast", nrow = 1e7, args=list(sparse = FALSE, parallel=FALSE))
    ),
    list(
      name = "(3) mclogit_fast(pll)",
      opt = list(fn="mclogit_fast", nrow = 1e7, args=list(sparse = FALSE, parallel=TRUE))
    )
  ))
  perf.dense.l <- run.repeated(1, "dense", list(
    list(
      name = "(2) mclogit_fast(ser)",
      opt = list(fn="mclogit_fast", nrow = 1e8, args=list(sparse = FALSE, parallel=FALSE))
    ),
    list(
      name = "(3) mclogit_fast(pll)",
      opt = list(fn="mclogit_fast", nrow = 1e8, args=list(sparse = FALSE, parallel=TRUE))
    )
  ))

  perf.dense <- rbind(
    perf.dense.s,
    perf.dense.m,
    perf.dense.l
  )
  fwrite(perf.dense, "perf.dense.csv")

  perf.dense.avg <- perf.dense[, .(
    "USER TIME[s]" = round(mean(USER), 3),
    "WALL TIME[s]" = round(mean(WALL), 3),
    "MAX RSS[MB]" = as.integer(mean(RSS_BYTE) / 1024 / 1024),
    "DATASET[MB]" = as.integer(mean(DATA_BYTE) / 1024 / 1024)
  ), by = .(NAME, NROW)]
  perf.dense.avg

  printMarkdownTable(perf.dense.avg)

  # --------------------------------------------------------------------------
  # sparse benchmarks
  # --------------------------------------------------------------------------
  perf.sparse.s <- run.repeated(repeats, "sparse", list(
    list(
      name = "(1) mclogit",
      opt = list(fn="mclogit", nrow = 1e6, args=list())
    ),
    list(
      name = "(2) mclogit_fast(ser)",
      opt = list(fn="mclogit_fast", nrow = 1e6, args=list(sparse = FALSE, parallel=FALSE))
    ),
    list(
      name = "(3) mclogit_fast(pll)",
      opt = list(fn="mclogit_fast", nrow = 1e6, args=list(sparse = FALSE, parallel=TRUE))
    ),
    list(
      name = "(4) mclogit_fast(pll,sparse)",
      opt = list(fn="mclogit_fast", nrow = 1e6, args=list(sparse = TRUE, parallel=TRUE))
    )
  ))
  perf.sparse.m <- run.repeated(repeats, "sparse", list(
    list(
      name = "(2) mclogit_fast(ser)",
      opt = list(fn="mclogit_fast", nrow = 1e7, args=list(sparse = FALSE, parallel=FALSE))
    ),
    list(
      name = "(3) mclogit_fast(pll)",
      opt = list(fn="mclogit_fast", nrow = 1e7, args=list(sparse = FALSE, parallel=TRUE))
    ),
    list(
      name = "(4) mclogit_fast(pll,sparse)",
      opt = list(fn="mclogit_fast", nrow = 1e7, args=list(sparse = TRUE, parallel=TRUE))
    )
  ))
  perf.sparse.l <- run.repeated(repeats, "sparse", list(
    list(
      name = "(4) mclogit_fast(pll,sparse)*",
      opt = list(fn="mclogit_fast", nrow = 1e8, args=list(sparse = TRUE, parallel=TRUE))
    )
  ))

  perf.sparse <- rbind(
    perf.sparse.s,
    perf.sparse.m,
    perf.sparse.l
  )
  fwrite(perf.sparse, "perf.sparse.csv")

  perf.sparse.avg <- perf.sparse[, .(
    "USER TIME[s]" = round(mean(USER), 3),
    "WALL TIME[s]" = round(mean(WALL), 3),
    "MAX RSS[MB]" = as.integer(mean(RSS_BYTE) / 1024 / 1024),
    "DATASET[MB]" = as.integer(mean(DATA_BYTE) / 1024 / 1024)
  ), by = .(NAME, NROW)]
  perf.sparse.avg

  printMarkdownTable(perf.sparse.avg)
}
