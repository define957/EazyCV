#' Symmetric Mean Absolute Percentage Error
#'
#' Compute the symmetric mean absolute percentage error (SMAPE) between the
#' observed values and the predicted values of a regression model. The
#' function follows the metric signature used by
#' \code{K_fold_cross_validation()}, so it can be passed directly through
#' \code{metrics}, or with extra arguments via \code{metrics_params}.
#'
#' @param y Observed values. A numeric vector or matrix.
#' @param y_hat Predicted values, in the same order and of the same length as
#'   \code{y}.
#'
#' @return A single numeric value: the mean of
#'   \code{2 * abs(y - y_hat) / (abs(y) + abs(y_hat))}, in the range
#'   \eqn{[0, 2]}. Lower values are better, and a perfect fit gives 0.
#'   Multiply by 100 to express it as a percentage.
#'
#' @details SMAPE measures the average percentage discrepancy between
#'   \code{y} and \code{y_hat}, scaled by the magnitude of both. Unlike MAPE
#'   it is symmetric in the two arguments, so over- and under-predictions of
#'   the same factor are penalized equally. A pair of zeros (\code{y} and
#'   \code{y_hat} both 0) has an undefined percentage error and contributes
#'   \code{NaN} to the mean; remove or impute such pairs before calling.
#'
#'   Both arguments must be numeric (integer or double); factors, character
#'   vectors, and logical vectors are rejected. Missing values are not
#'   allowed: \code{NA} in \code{y} or \code{y_hat} raises an error. The two
#'   arguments must have the same, non-zero length, otherwise an error is
#'   raised. Matrix inputs are handled element-wise, and a matrix \code{y} can
#'   be paired with a vector \code{y_hat} of matching length.
#'
#' @examples
#' y     <- c(3, -0.5, 2, 7)
#' y_hat <- c(2.5, 0.0, 2, 8)
#'
#' symmetric_mean_absolute_percentage_error(y, y_hat)
#' symmetric_mean_absolute_percentage_error(y, y_hat) * 100  # as a percentage
#'
#' # Use as a metric in cross-validation
#' \dontrun{
#' my_lm <- function(X, y) {
#'   lm.fit(cbind(1, X), y)
#' }
#'
#' my_predict <- function(fit, X) {
#'   cbind(1, X) %*% fit$coefficients
#' }
#'
#' cv <- K_fold_cross_validation(
#'   model = my_lm, X = X, y = y, folds = folds,
#'   metrics = list(smape = symmetric_mean_absolute_percentage_error),
#'   predict_func = my_predict
#' )
#' }
#'
#' @seealso \code{\link{K_fold_cross_validation}}, \code{\link{mean_absolute_error}},
#'   \code{\link{mean_squared_error}}, \code{\link{root_mean_squared_error}}
#' @author Zhang Jiaqi
#' @export
symmetric_mean_absolute_percentage_error <- function(y, y_hat) {
  if (!is.numeric(y) || !is.numeric(y_hat)) {
    stop("'y' and 'y_hat' must be numeric.")
  }
  if (length(y) != length(y_hat)) {
    stop(sprintf(
      "'y' and 'y_hat' must have the same length (got %d and %d).",
      length(y), length(y_hat)
    ))
  }
  if (length(y) == 0L) {
    stop("'y' and 'y_hat' must not be empty.")
  }
  if (anyNA(y) || anyNA(y_hat)) {
    stop("'y' and 'y_hat' must not contain missing values.")
  }
  return(mean(2 * abs(y - y_hat) / (abs(y) + abs(y_hat))))
}
