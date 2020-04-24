mclogit_fast <- function(formula,
                         data=parent.frame(),
                         random=NULL,
                         subset,
                         weights=NULL,
                         offset=NULL,
                         na.action = getOption("na.action"),
                         model = TRUE, x = FALSE, y = TRUE,
                         contrasts=NULL,
                         start=NULL,
                         parallel=TRUE,
                         sparse=FALSE,
                         verbose=FALSE,
                         ...
                    ){
# Assumptions:
#   left hand side of formula: cbind(counts,  choice set index)
#   right hand side of the formula: attributes
#   intercepts are removed!

    call <- match.call(expand.dots = TRUE)

    if(missing(data)) data <- environment(formula)
    mf <- match.call(expand.dots = FALSE)
    m <- match(c("formula", "data", "subset", "weights", "offset", "na.action"), names(mf), 0)
    mf <- mf[c(1, m)]
    mf$drop.unused.levels <- TRUE
    mf[[1]] <- as.name("model.frame")

    stopifnot(is.null(random))

    mf <- eval(mf, parent.frame())
    gc()

    mt <- terms(formula)
    na.action <- attr(mf,"na.action")
    weights <- as.vector(model.weights(mf))
    if(!is.null(weights) && !is.numeric(weights))
        stop("'weights' must be a numeric vector")

    Y <- as.matrix(model.response(mf, "any"))

    if(ncol(Y)<2) stop("need response counts and choice set indicators")
    sets <- Y[,2]
    Y <- Y[,1]
    stopifnot(!is.unsorted(sets))

    if (is.null(weights)) {
      weights <- rep(1., length(Y))
    }

    if(isTRUE(sparse)) {
      cat("use sparse\n")
      X <- my.sparse.model.matrix(mt,mf,contrasts,verbose=verbose)
      contrasts <- attr(X, "contrasts")
      xlevels <- .getXlevels(mt,mf)
      icpt <- match("(Intercept)",colnames(X),nomatch=0)
      if(icpt) X <- X[,-icpt,drop=FALSE]
      gc()

      wsum <- wsum_per_col_sparse(X, Y, weights)
      fit <- mclogit_fast_sparse_fit_pll(Y, sets, weights, X)
      fit$wsum <- wsum

    } else {
      cat("use dense\n")
      X <- model.matrix(mt,mf,contrasts)
      contrasts <- attr(X, "contrasts")
      xlevels <- .getXlevels(mt,mf)
      icpt <- match("(Intercept)",colnames(X),nomatch=0)
      if(icpt) X <- X[,-icpt,drop=FALSE]

      gc()
      if(isTRUE(parallel)) {
        fit <- mclogit_fast_dense_fit_pll(Y, sets, weights, X)
      } else {
        fit <- mclogit_fast_dense_fit(Y, sets, weights, X)
      }
    }

    names(fit$coef) <- colnames(X)
    rownames(fit$covmat) <- colnames(X)
    colnames(fit$covmat) <- colnames(X)

    fit <- c(fit, list(call = call, formula = formula,
                       terms = mt,
                       data = data, N = nrow(X),
                       contrasts = contrasts,
                       xlevels = xlevels,
                       na.action = na.action,
                       weights = weights,
                       df.residual = nrow(X) - ncol(X),
                       model.df = ncol(X),
                       model = mf))

    class(fit) <- c("mclogit_fast")
    return(fit)
}


summary.mclogit_fast <- function(object,dispersion=NULL,correlation = FALSE, symbolic.cor = FALSE,...){
  ## calculate coef table
  coef <- object$coef
  covmat.scaled <- object$covmat
  var.cf <- diag(covmat.scaled)
  s.err <- sqrt(var.cf)
  zvalue <- coef/s.err
  pvalue <- 2*pnorm(-abs(zvalue))

  coef.table <- array(NA,dim=c(length(coef),4))
  dimnames(coef.table) <- list(names(coef),
                               c("Estimate", "Std. Error","z value","Pr(>|z|)"))
  coef.table[,1] <- coef
  coef.table[,2] <- s.err
  coef.table[,3] <- zvalue
  coef.table[,4] <- pvalue

  ans <- c(object[c("call","terms","deviance","contrasts",
                    "null.deviance","iter","na.action","model.df",
                    "df.residual","N","converged")],
           list(coef = coef.table,
                cov.coef=object$covmat))
  p <- length(coef)
  if(correlation && p > 0) {
    dd <- sqrt(diag(ans$cov.coef))
    ans$correlation <-
      ans$cov.coef/outer(dd,dd)
    ans$symbolic.cor <- symbolic.cor
  }
  class(ans) <- "summary.mclogit_fast"
  return(ans)
}

print.summary.mclogit_fast <-
  function (x, digits = max(3, getOption("digits") - 3),
            symbolic.cor = x$symbolic.cor,
            signif.stars = getOption("show.signif.stars"), ...){
    cat("\nCall:\n")
    cat(paste(deparse(x$call), sep="\n", collapse="\n"), "\n\n", sep="")

    coefs <- x$coef
    printCoefmat(coefs, digits=digits, signif.stars=signif.stars,
                 na.print="NA", ...)

    cat("\nNull Deviance:     ", format(signif(x$null.deviance, digits)),
        "\nResidual Deviance: ", format(signif(x$deviance, digits)),
        "\nNumber of Fisher Scoring iterations: ", x$iter,
        "\nNumber of observations: ",x$N,
        "\n")
    correl <- x$correlation
    if(!is.null(correl)) {
      p <- NCOL(correl)
      if(p > 1) {
        cat("\nCorrelation of Coefficients:\n")
        if(is.logical(symbolic.cor) && symbolic.cor) {
          print(symnum(correl, abbr.colnames = NULL))
        } else {
          correl <- format(round(correl, 2), nsmall = 2, digits = digits)
          correl[!lower.tri(correl)] <- ""
          print(correl[-1, -p, drop=FALSE], quote = FALSE)
        }
      }
    }

    if(!x$converged) cat("\n\nNote: Algorithm did not converge.\n")
    if(nchar(mess <- naprint(x$na.action))) cat("  (",mess, ")\n\n", sep="")
    else cat("\n\n")
    invisible(x)
  }

vcov.mclogit_fast <- function(object,...){
  return(object$covmat)
}

coef.mclogit_fast <- function(object,...){
  return(object$coef)
}

# c&p from stats::confint.lm
confint.mclogit_fast <- function (object, parm, level = 0.95, ...)
{
  cf <- coef(object)
  ses <- sqrt(diag(vcov(object)))
  pnames <- names(ses)
  if (is.matrix(cf))
    cf <- setNames(as.vector(cf), pnames)
  if (missing(parm))
    parm <- pnames
  else if (is.numeric(parm))
    parm <- pnames[parm]
  a <- (1 - level)/2
  a <- c(a, 1 - a)
  fac <- qt(a, object$df.residual)
  pct <- stats:::format.perc(a, 3) # format.prec is not exported from stats
  ci <- array(NA_real_, dim = c(length(parm), 2L), dimnames = list(parm,
                                                                   pct))
  ci[] <- cf[parm] + ses[parm] %o% fac
  ci
}

strip.mclogit_fast <- function(object) {
  keep <- c("call", "formula", "contrasts", "xlevels", "coef", "covmat", "N", 
            "null.deviance", "deviance", "converged", "na.action", "df.residual",
            "model.df", "ll")
  if("wsum" %in% names(object)) {
    keep <- c(keep, "wsum")
  }
  object.stripped <- object[keep]
  class(object.stripped) <- class(object)
  return(object.stripped)
}

AIC.mclogit_fast <- function(object, k) {
  -2 * object$ll + k * object$model.df
}
