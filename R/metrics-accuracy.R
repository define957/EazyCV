#' Classification Accuracy
#'
#' Compute the proportion of predictions that exactly match the ground
#' truth. The function follows the metric signature used by
#' \code{K_fold_cross_validation()}, so it can be passed directly through
#' \code{metrics}, or with extra arguments via \code{metrics_params}.
#'
#' @param y Ground truth labels. A vector, matrix, or factor; matrices are
#'   flattened in column-major order.
#' @param y_hat Predicted labels, in the same order and of the same length as
#'   \code{y}.
#' @param na.rm Logical; if \code{TRUE}, pairs containing a missing value in
#'   either \code{y} or \code{y_hat} are discarded before averaging. Default
#'   is \code{FALSE}.
#'
#' @return A single numeric value between 0 and 1: the fraction of matching
#'   predictions.
#'
#' @details Comparisons are made with \code{==}. Factors are converted to
#'   character first, so integer level codes are never compared against label
#'   strings. Numeric predictions are compared against character labels after
#'   coercion to character.
#'
#' @examples
#' y <- c("a", "b", "a", "b")
#' y_hat <- c("a", "b", "b", "b")
#' accuracy(y, y_hat)
#'
#' # Use as a metric in cross-validation
#' \dontrun{
#' cv <- K_fold_cross_validation(
#'   model = my_model, X = X, y = y, folds = folds,
#'   metrics = list(acc = accuracy), predict_func = my_predict
#' )
#' }
#'
#' @author Zhang Jiaqi
#' @seealso \code{\link{K_fold_cross_validation}}
#' @export
accuracy <- function(y, y_hat, na.rm = FALSE) {

  y     <- if (is.factor(y)) as.character(y) else as.vector(y)
  y_hat <- if (is.factor(y_hat)) as.character(y_hat) else as.vector(y_hat)

  if (length(y) != length(y_hat)) {
    stop(sprintf(
      "'y' and 'y_hat' must have the same length (got %d and %d).",
      length(y), length(y_hat)
    ))
  }
  if (length(y) == 0L) {
    stop("'y' and 'y_hat' must not be empty.")
  }

  return(mean(y == y_hat, na.rm = na.rm))

}
