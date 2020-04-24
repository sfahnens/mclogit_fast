my.sparse.model.matrix <- function (object, data = environment(object), contrasts.arg = NULL, 
  xlev = NULL, transpose = FALSE, drop.unused.levels = FALSE, 
  row.names = TRUE, verbose = FALSE, ...) 
{
  t <- if (missing(data)) 
    terms(object)
  else terms(object, data = data)
  if (is.null(attr(data, "terms"))) 
    data <- model.frame(object, data, xlev = xlev)
  else {
    reorder <- match(sapply(attr(t, "variables"), deparse, 
      width.cutoff = 500)[-1L], names(data))
    if (anyNA(reorder)) 
      stop("model frame and formula mismatch in model.matrix()")
    if (!Matrix:::isSeq(reorder, ncol(data), Ostart = FALSE)) 
      data <- data[, reorder, drop = FALSE]
  }
  int <- attr(t, "response")
  if (length(data)) {
    contr.funs <- as.character(getOption("contrasts"))
    namD <- names(data)
    for (i in namD) if (is.character(data[[i]])) 
      data[[i]] <- factor(data[[i]])
    isF <- vapply(data, function(x) is.factor(x) || is.logical(x), 
      NA)
    isF[int] <- FALSE
    isOF <- vapply(data, is.ordered, NA)
    for (nn in namD[isF]) if (is.null(attr(data[[nn]], "contrasts"))) 
      my.contrasts(data[[nn]]) <- contr.funs[1 + isOF[nn]]
    if (!is.null(contrasts.arg) && is.list(contrasts.arg)) {
      if (is.null(namC <- names(contrasts.arg))) 
        stop("invalid 'contrasts.arg' argument")
      for (nn in namC) {
        if (is.na(ni <- match(nn, namD))) 
          warning(gettextf("variable '%s' is absent, its contrast will be ignored", 
            nn), domain = NA)
        else {
          ca <- contrasts.arg[[nn]]
          if (is.matrix(ca)) 
            my.contrasts(data[[ni]], ncol(ca)) <- ca
          else my.contrasts(data[[ni]]) <- contrasts.arg[[nn]]
        }
      }
    }
  }
  else {
    isF <- FALSE
    data <- cbind(data, x = 0)
  }
  if (verbose) {
    cat("model.spmatrix(t, data, ..)  with t =\n")
    str(t, give.attr = FALSE)
  }
  ans <- my.model.spmatrix(t, data, transpose = transpose, drop.unused.levels = drop.unused.levels, 
    row.names = row.names, verbose = verbose)
  attr(ans, "contrasts") <- lapply(data[isF], function(x) attr(x, 
    "contrasts"))
  ans
}

`my.contrasts<-` <- function (x, how.many, value) 
{
  if (is.logical(x)) {
    x <- as.integer(x) + 1L
    levels(x) <- c("FALSE", "TRUE")
    class(x) <- "factor"
  }
  if (!is.factor(x)) 
    stop("contrasts apply only to factors")
  if (nlevels(x) < 2L) 
    stop("contrasts can be applied only to factors with 2 or more levels")
  if (is.function(value)) 
    value <- value(nlevels(x))
  if ((is.n <- is.numeric(value)) || (isS4(value) && methods::is(value, 
    "Matrix"))) {
    if (is.n) 
      value <- as.matrix(value)
    nlevs <- nlevels(x)
    if (nrow(value) != nlevs) 
      stop("wrong number of contrast matrix rows")
    n1 <- if (missing(how.many)) 
      nlevs - 1L
    else how.many
    nc <- ncol(value)
    rownames(value) <- levels(x)
    if (nc < n1) {
      if (!is.n) 
        value <- as.matrix(value)
      cm <- qr(cbind(1, value))
      if (cm$rank != nc + 1) 
        stop("singular contrast matrix")
      cm <- qr.qy(cm, diag(nlevs))[, 2L:nlevs]
      cm[, 1L:nc] <- value
      dimnames(cm) <- list(levels(x), NULL)
      if (!is.null(nmcol <- dimnames(value)[[2L]])) 
        dimnames(cm)[[2L]] <- c(nmcol, rep.int("", n1 - 
          nc))
    }
    else cm <- value[, 1L:n1, drop = FALSE]
  }
  else if (is.character(value)) 
    cm <- value
  else if (is.null(value)) 
    cm <- NULL
  else stop("numeric contrasts or contrast name expected")
  attr(x, "contrasts") <- cm
  x
}



my.model.spmatrix <- function (trms, mf, transpose = FALSE, drop.unused.levels = FALSE, 
  row.names = TRUE, verbose = FALSE) 
{
  stopifnot(is.data.frame(mf))
  n <- nrow(mf)
  if (row.names) 
    rnames <- row.names(mf)
  fnames <- names(mf <- unclass(mf))
  attributes(mf) <- list(names = fnames)
  if (length(factorPattern <- attr(trms, "factors"))) {
    d <- dim(factorPattern)
    nVar <- d[1]
    nTrm <- d[2]
    n.fP <- dimnames(factorPattern)
    fnames <- n.fP[[1]]
    Names <- n.fP[[2]]
  }
  else {
    nVar <- nTrm <- 0L
    fnames <- Names <- character(0)
  }
  stopifnot((m <- length(mf)) >= nVar)
  if (verbose) 
    cat(sprintf("model.spm..(): (n=%d, nVar=%d (m=%d), nTrm=%d)\n", 
      n, nVar, m, nTrm))
  if (m > nVar) 
    mf <- mf[seq_len(nVar)]
  stopifnot(fnames == names(mf))
  noVar <- nVar == 0
  is.f <- if (noVar) 
    logical(0)
  else vapply(mf, function(.) is.factor(.) | is.logical(.), 
    NA)
  indF <- which(is.f)
  if (verbose) {
    cat(" --> indF =\n")
    print(indF)
  }
  hasInt <- attr(trms, "intercept") == 1
  if (!hasInt && length(indF)) {
    if (any(i1 <- factorPattern[indF, ] == 1)) 
      factorPattern[indF, ][which.max(i1)] <- 2L
    else {
    }
  }
  f.matr <- structure(vector("list", length = length(indF)), 
    names = fnames[indF])
  i.f <- 0
  for (i in seq_len(nVar)) {
    nam <- fnames[i]
    f <- mf[[i]]
    if (is.f[i]) {
      fp <- factorPattern[i, ]
      contr <- attr(f, "contrasts")
      f.matr[[(i.f <- i.f + 1)]] <- lapply(my.fac2Sparse(f, 
        to = "d", drop.unused.levels = drop.unused.levels, 
        factorPatt12 = 1:2 %in% fp, contrasts.arg = contr), 
        function(s) {
          if (is.null(s)) 
            return(s)
          rownames(s) <- paste0(nam, if (is.null(rownames(s))) 
            seq_len(nrow(s))
          else rownames(s))
          s
        })
    }
    else {
      if (any(iA <- (cl <- class(f)) == "AsIs")) 
        class(f) <- if (length(cl) > 1L) 
          cl[!iA]
      nr <- if (is.matrix(f)) 
        nrow(f <- t(f))
      else (dim(f) <- c(1L, length(f)))[1]
      if (is.null(rownames(f))) 
        rownames(f) <- if (nr == 1) 
          nam
        else paste0(nam, seq_len(nr))
      mf[[i]] <- f
    }
  }
  if (verbose) {
    cat(" ---> f.matr list :\n")
    str(f.matr, max = as.integer(verbose))
    fNms <- format(dQuote(Names))
    dim.string <- gsub("5", as.character(floor(1 + log10(n))), 
      " -- concatenating (r, rj): dim = (%5d,%5d) | (%5d,%5d)\n")
  }
  getR <- function(N) if (!is.null(r <- f.matr[[N]])) 
    r[[factorPattern[N, nm]]]
  else mf[[N]]
  vNms <- "(Intercept)"[hasInt]
  counts <- integer(nTrm)
  r <- if (hasInt) 
    new("dgCMatrix", i = 0:(n - 1L), p = c(0L, n), Dim = c(n, 
      1L), x = rep.int(1, n))
  else new("dgCMatrix", Dim = c(n, 0L))
  if (transpose) 
    r <- t(r)
  iTrm <- seq_len(nTrm)
  for (j in iTrm) {
    nm <- Names[j]
    if (verbose) 
      cat(sprintf("term[%2d] %s .. ", j, fNms[j]))
    nmSplits <- strsplit(nm, ":", fixed = TRUE)[[1]]
    rj <- my.sparseInt.r(lapply(nmSplits, getR), do.names = TRUE, 
      forceSparse = TRUE, verbose = verbose)
    if (verbose) 
      cat(sprintf(dim.string, nrow(r), ncol(r), nrow(rj), 
        ncol(rj)))
    r <- if (transpose) 
      .Call("Csparse_vertcat", r, rj, PACKAGE = "Matrix")
    else .Call("Csparse_horzcat", r, Matrix:::t(rj), PACKAGE = "Matrix")
    vNms <- c(vNms, dimnames(rj)[[1]])
    counts[j] <- nrow(rj)
  }
  rns <- if (row.names) 
    rnames
  dimnames(r) <- if (transpose) 
    list(vNms, rns)
  else list(rns, vNms)
  attr(r, "assign") <- c(if (hasInt) 0L, rep(iTrm, counts))
  r
}

my.fac2Sparse <- function (from, to = c("d", "i", "l", "n", "z"), drop.unused.levels = TRUE, 
  giveCsparse = TRUE, factorPatt12, contrasts.arg = NULL) 
{
  stopifnot(is.logical(factorPatt12), length(factorPatt12) == 2)
  if (any(factorPatt12)) {
    if(to == "d" && drop.unused.levels == FALSE && giveCsparse == TRUE) {
      m <- fac_to_sparse(as.integer(from), length(levels(from)))
      m@Dimnames <- list(levels(from), names(from))
    } else {
      m <- Matrix::fac2sparse(from, to = to, drop.unused.levels = drop.unused.levels, 
        giveCsparse = giveCsparse)
    }
  }

  ans <- list(NULL, if (factorPatt12[2]) m)
  if (factorPatt12[1]) {
    if (is.null(contrasts.arg)) 
      contrasts.arg <- getOption("contrasts")[if (is.ordered(from)) 
        "ordered"
      else "unordered"]
    ans[[1]] <- Matrix::crossprod(if (is.character(contrasts.arg)) {
      stopifnot(is.function(FUN <- get(contrasts.arg)))
      FUN(rownames(m), sparse = TRUE)
    }
    else as(contrasts.arg, "sparseMatrix"), m)
  }
  ans
}

my.sparseInt.r <- function (rList, do.names = TRUE, forceSparse = FALSE, verbose = FALSE) 
{
  nl <- length(rList)
  if (forceSparse) 
    F <- function(m) if (is.matrix(m)) 
      .Call("dense_to_Csparse", m, PACKAGE = "Matrix")
    else m
  if (verbose) 
    cat("sparseInt.r(<list>[1:", nl, "], f.Sp=", forceSparse, 
      "): is.mat()= (", paste(symnum(vapply(rList, is.matrix, 
        NA)), collapse = ""), ")\n", sep = "")
  if (nl == 1) {
    if (forceSparse) 
      F(rList[[1]])
    else rList[[1]]
  }
  else {
    # DEBUG_RLIST <<- copy(rList)
    r <- rList[[1]]
    for (j in 2:nl) r <- my.sparse2int(r, rList[[j]], do.names = do.names, 
      verbose = verbose)
    if (forceSparse) 
      F(r)
    else r
  }
}

my.sparse2int <- function (X, Y, do.names = TRUE, forceSparse = FALSE, verbose = FALSE) 
{
  if (do.names) {
    dnx <- dimnames(X)
    dny <- dimnames(Y)
  }
  dimnames(Y) <- dimnames(X) <- list(NULL, NULL)
  nx <- nrow(X)
  ny <- nrow(Y)
  
  stopifnot(nx > 0 && ny > 0)
   
  r <- if ((nX <- is.numeric(X)) | (nY <- is.numeric(Y))) {
    if (nX) {
      if (nY || nx > 1) {
        F <- if (forceSparse) function(m) .Call("dense_to_Csparse", m, PACKAGE = "Matrix") else identity
        
        F((if (ny == 1) X else X[rep.int(seq_len(nx), ny), ]) * 
          (if (nx == 1) Y else Y[rep(seq_len(ny), each = nx), ]))
      } else {
        r <- Y
        dp <- Y@p[-1] - Y@p[-(Y@Dim[2] + 1L)]
        r@x <- X[dp == 1L] * Y@x
        r
      }
    } else {
      if (ny == 1) {
        r <- X
        dp <- X@p[-1] - X@p[-(X@Dim[2] + 1L)]
        r@x <- Y[dp == 1L] * X@x
        r
      } else {
        X[rep.int(seq_len(nx), ny), ] * 
        (if (nx == 1) Y else Y[rep(seq_len(ny), each = nx), ])
      }
    }
  } else {
    if(class(X) == "dgCMatrix" && class(Y) == "dgCMatrix") {
      matrix_cols(X, Y)
    } else { # what is it?
      (if (ny == 1) X else X[rep.int(seq_len(nx), ny), ]) *
      (if (nx == 1) Y else Y[rep(seq_len(ny), each = nx), ])
    }
  }
  if (verbose) 
    cat(sprintf(" sp..2int(%s[%d],%s[%d]) ", if (nX) 
      "<N>"
    else "<sparse>", nx, if (nY) 
      "<N>"
    else "<sparse>", ny))
  if (do.names) {
    if (!is.null(dim(r)) && !is.null(nX <- dnx[[1]]) && 
      !is.null(nY <- dny[[1]])) 
      rownames(r) <- outer(nX, nY, paste, sep = ":")
  }
  r
}
