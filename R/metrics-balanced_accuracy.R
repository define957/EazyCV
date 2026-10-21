#' Balanced Accuracy
#'
#' Compute balanced accuracy for classification, which is the average of
#' recall obtained on each class.
#' @author Zhang Jiaqi
#' @param y,y_hat real values and fitted values.
#' @param sample_weight numeric vector of sample weights with one value per
#'   observation, default \code{NULL}.
#' @param adjusted if \code{TRUE}, adjust the score for chance so that random
#'   performance scores 0 and perfect performance scores 1.
#' @return balanced accuracy score, a numeric scalar.
#' @details Missing values are not allowed: \code{NA} in \code{y},
#'   \code{y_hat}, or \code{sample_weight} raises an error. This matches the
#'   input validation of \code{sklearn.metrics.balanced_accuracy_score()}.
#' @export
balanced_accuracy <- function(y, y_hat, sample_weight = NULL, adjusted = FALSE) {
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
  if (!is.null(sample_weight)) {
    if (!is.numeric(sample_weight) || length(sample_weight) != length(y)) {
      stop(
        "'sample_weight' must be a numeric vector with one value per observation."
      )
    }
    if (anyNA(sample_weight)) {
      stop("'sample_weight' must not contain missing values.")
    }
  }
  y        <- as.character(y)
  y_hat    <- as.character(y_hat)
  labels   <- sort(union(unique(y), unique(y_hat)))
  n_labels <- length(labels)
  f_y      <- factor(y, levels = labels)
  f_yhat   <- factor(y_hat, levels = labels)

  # Get Confusion Matrix: rows = true class, cols = predicted class.
  if (is.null(sample_weight)) {
    cm <- table(f_y, f_yhat)
  } else {
    cm <- tapply(sample_weight, list(f_y, f_yhat), sum)
    cm[is.na(cm)] <- 0
  }
  cm <- unname(cm)

  # Per-Class Recall: empty rows become NA instead of 0/0.
  row_sum                 <- rowSums(cm)
  per_class_recall        <- rep(NA_real_, n_labels)
  valid                   <- row_sum > 0
  per_class_recall[valid] <- diag(cm)[valid] / row_sum[valid]

  # Drop Empty Classes.
  if (any(!valid)) {
    warning("y_hat contains classes not in y")
    per_class_recall <- per_class_recall[valid]
  }

  score <- mean(per_class_recall)

  # Chance Correction: (score - 1/k) / (1 - 1/k), k = surviving classes.
  if (adjusted) {
    k <- length(per_class_recall)
    if (k == 0) { stop("division by zero: all classes were dropped") }
    score <- (score - 1 / k) / (1 - 1 / k)
  }
  score
}
