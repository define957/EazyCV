#' Root Mean Squared Error
#'
#' Compute the root mean squared error (RMSE) between the observed values and
#' the predicted values of a regression model. The function follows the metric
#' signature used by \code{K_fold_cross_validation()}, so it can be passed
#' directly through \code{metrics}, or with extra arguments via
#' \code{metrics_params}.
#'
#' @param y Observed values. A numeric vector or matrix.
#' @param y_hat Predicted values, in the same order and of the same length as
#'   \code{y}.
#'
#' @return A single numeric value: the square root of the average of the
#'   squared residuals, \code{sqrt(mean((y - y_hat)^2))}.
#'
#' @details RMSE is the square root of the
#'   \code{\link{mean_squared_error}}: \code{sqrt(mean_squared_error(y,
#'   y_hat))}. Unlike MSE, which is expressed in the squared units of
#'   \code{y}, RMSE keeps the original units of \code{y}, so it is directly
#'   comparable to the observed values. It is still sensitive to outliers,
#'   since large residuals dominate after squaring. Lower values are better,
#'   and a perfect fit gives 0.
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
#' root_mean_squared_error(y, y_hat)
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
#'   metrics = list(rmse = root_mean_squared_error), predict_func = my_predict
#' )
#' }
#'
#' @seealso \code{\link{mean_squared_error}},
#'   \code{\link{K_fold_cross_validation}}, \code{\link{accuracy}},
#'   \code{\link{balanced_accuracy}}
#' @author Zhang Jiaqi
#' @export
root_mean_squared_error <- function(y, y_hat) {
  return(sqrt(mean_squared_error(y, y_hat)))
}
