byRow <- function(dt, expr) {
  sexpr <- substitute(expr)
  
  if(nrow(dt) == 0) {
    return(invisible(NULL))
  }
  
  parent <- parent.frame()
  ilapply(1: nrow(dt), function(i) {
    e <- new.env(parent = parent)
    byRow.populateEnv(dt, i, e)
    
    eval(sexpr, envir = e)
    NULL
  })
}

byRow.f <- function(dt, expr) {
  sexpr <- substitute(expr)
  
  if(nrow(dt) == 0) {
    return(data.table())
  }
  
  parent <- parent.frame()
  return(flapply(1: nrow(dt), function(i) {
    e <- new.env(parent = parent)
    byRow.populateEnv(dt, i, e)
    return(eval(sexpr, envir = e))
  }))
}

byRow.populateEnv <- function(dt, i, env = parent.frame()) {
  for(n in colnames(dt)) {
    assign(n, dt[[n]][[i]], envir = env)
    assign(paste0("r.",n), dt[[n]][[i]], envir = env)
  }
}

flapply <- function(X, FUN, ...)
  flatten(lapply(X, FUN, ...))

flatten <- function(lst) {
  maybeUnpack <- function(e) {
    if ("list" %in% class(e)) {
      return(flatten(e))
    } else {
      return(e)
    }
  }
  return(rbindlist(lapply(lst, maybeUnpack), fill = TRUE))
}

cm.summary <- function(m, ...) {
  sm <- mclogit_fast::summary.mclogit_fast(m)
  printCoefmat(sm$coef, digits=3, signif.stars=TRUE, na.print="NA", ...)
  cat(sprintf("values:\n  tt/px %.2f  ic/px %.2f  ic/tt %.2f\n",
              60 * m$coef["TT"] / m$coef["PX"],
              m$coef["IC"] / m$coef["PX"],
              m$coef["IC"] / m$coef["TT"]))
}

value.ratio <- function(m, coef.a, coef.b, mult = 1) {
  # confidence intervals via Monte-Carlo simulation
  coef <- m$coef[c(coef.a, coef.b)]
  covmat <- m$covmat[which(rownames(m$covmat) %in% c(coef.a, coef.b)), 
                     which(colnames(m$covmat) %in% c(coef.a, coef.b))]
  
  sim <- MASS::mvrnorm(n = 100000, coef, covmat)
  q <- quantile(sim[, 1] / sim[, 2], c(.025, .975), names = FALSE)
  return(data.table(#LABEL = sprintf("%s/%s", coef.a, coef.b),
    COEF = coef[[1]] / coef[[2]] * mult, 
    Q_MIN = q[[1]] * mult, Q_MAX = q[[2]] * mult))
}

value.ratio.interacted <- function(m, suffix.a, suffix.b, 
                                   label = sprintf("%s/%s", suffix.a, suffix.b), mult = 1) {
  dat <- data.table(NAME = names(m$coef), COEF = m$coef)
  
  # thanks, stackoverflow: https://stackoverflow.com/a/24938928
  dat[, c("PREFIX", "SUFFIX") := tstrsplit(NAME, ":\\s*(?=[^:]+$)", perl=TRUE)]
  
  dat <- merge(dat[SUFFIX == suffix.a], dat[SUFFIX == suffix.b], by = "PREFIX")
  
  # byRow.populateEnv(vot, 1)
  vr <- byRow.f(dat, {
    cbind(data.table(LABEL = label, PREFIX = r.PREFIX), 
          value.ratio(m, coef.a = r.NAME.x, coef.b = r.NAME.y, mult = mult))
  })
  return(vr)
}
