scaler_check <- function(X) {

  if (is.null(dim(X))) {
    X <- as.matrix(X)
  }
  scaler_checker <- list(
    X = X,
    num_features = ncol(X)
  )

  return(scaler_checker)

}

trans_check <- function(scaler, X) {

  if (is.null(dim(X))) {
    X <- as.matrix(X)
  }
  X_features <- ncol(X)
  if (X_features != scaler$num_features) {
    stop("X must have the same number of features as the scaler.")
  }
  trans_checker <- list(
    X = X,
    num_samples = nrow(X),
    num_features = ncol(X)
  )

  return(trans_checker)

}

#' Scaler Transform Function
#'
#' Generic function that applies a fitted scaler/transformer to a data
#' matrix. Methods exist for scaler objects produced by
#' \code{standard_scaler()}; used internally by the preprocessing
#' pipelines of \code{K_fold_cross_validation()}.
#'
#' @author Zhang Jiaqi
#' @param scaler A scaler object (e.g., from \code{standard_scaler()}).
#' @param X Data matrix to transform; coerced with \code{as.matrix()} if
#'   needed. Must have the same number of features as the data the scaler
#'   was fitted on.
#' @return The transformed matrix (same dimensions as \code{X}).
#' @examples
#' scaler <- standard_scaler(mtcars[, c("wt", "disp")])
#' trans(scaler, mtcars[, c("wt", "disp")])
#' @seealso \code{\link{standard_scaler}}
#' @export
trans <- function(scaler, X) {

  UseMethod("trans")

}

#' Standard Scaler
#'
#' Fit a standard scaler (per-feature centering and scaling) on a data
#' matrix. The fitted object can then be applied to other matrices with
#' the same number of features via \code{trans()}.
#'
#' @author Zhang Jiaqi
#' @param X Data matrix (or vector) used to compute the per-feature mean
#'   and standard deviation.
#' @return An object of class \code{standard_scaler}: a list with
#'   components \code{X_mean} (per-feature means), \code{X_sd} (per-feature
#'   standard deviations), and \code{num_features}.
#' @details Features with zero standard deviation are centered only (not
#'   divided), so constant columns stay constant instead of producing
#'   \code{NaN}.
#' @examples
#' scaler <- standard_scaler(mtcars[, c("wt", "disp")])
#' scaler$X_mean
#' trans(scaler, mtcars[, c("wt", "disp")])
#' @seealso \code{\link{trans}}
#' @export
standard_scaler <- function(X) {

  scaler_checker <- scaler_check(X)
  X              <- scaler_checker$X
  X_mean         <- colMeans(X)
  X_sd           <- apply(X, 2, sd)

  result <- list(
    X_mean = X_mean,
    X_sd = X_sd,
    num_features = scaler_checker$num_features
  )

  class(result) <- "standard_scaler"

  return(result)

}

#' @export
trans.standard_scaler <- function(scaler, X) {

  trans_checker <- trans_check(scaler, X)
  X             <- trans_checker$X
  X_scaled      <- matrix(0, trans_checker$num_samples, trans_checker$num_features)
  X_mean        <- scaler$X_mean
  X_sd          <- scaler$X_sd
  idx           <- which(X_sd != 0)

  # Calculate centered X
  X_centered_T     <- (t(X) - X_mean)
  # Calculate Standardized X (sd != 0)
  X_scaled[, idx]  <- t(X_centered_T[idx, ] / X_sd[idx])
  X_scaled[, -idx] <- t(X_centered_T[-idx, ])

  return(X_scaled)

}
