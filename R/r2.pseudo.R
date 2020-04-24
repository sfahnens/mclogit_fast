r2.pseudo <- function(m) {
  #           dev = 2 * (logLik(y) - logLik(y_hat)) // Lik(y) = 1 -> logLik(y) = 0
  #           dev = -2 * logLik(y_hat)
  # logLik(y_hat) = -0.5 * dev

  # mc_fadden's r2 = 1 - (logLik(y_hat) - K) / logLik(y_hat, 0)
  return(1 - ((-0.5 * m$deviance) - length(coef(m))) / (-0.5 * m$null.deviance))
}
