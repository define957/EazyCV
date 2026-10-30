#' Mean Absolute Error
#'
#' Compute the mean absolute error (MAE) between the observed values and the
#' predicted values of a regression model. The function follows the metric
#' signature used by \code{K_fold_cross_validation()}, so it can be passed
#' directly through \code{metrics}, or with extra arguments via
#' \code{metrics_params}.
#'
#' @param y Observed values. A numeric vector or matrix.
#' @param y_hat Predicted values, in the same order and of the same length as
#'   \code{y}.
#'
#' @return A single numeric value: the average of the absolute residuals,
#'   \code{mean(abs(y - y_hat))}.
#'
#' @details MAE is expressed in the original units of \code{y} and grows
#'   linearly with the residual size, so every error contributes in proportion
#'   to its magnitude and outliers weigh less heavily than in the mean
#'   squared error. Lower values are better, and a perfect fit gives 0.
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
#' mean_absolute_error(y, y_hat)
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
#'   metrics = list(mae = mean_absolute_error), predict_func = my_predict
#' )
#' }
#'
#' @seealso \code{\link{K_fold_cross_validation}},
#'   \code{\link{mean_squared_error}}, \code{\link{root_mean_squared_error}},
#'   \code{\link{accuracy}}, \code{\link{balanced_accuracy}}
#' @author Zhang Jiaqi
#' @export
mean_absolute_error <- function(y, y_hat) {
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
  return(mean(abs(y - y_hat)))
}
