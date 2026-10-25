#' Geometric Mean of Recall
#'
#' Compute the geometric mean of per-class recall.
#' @author Zhang Jiaqi
#' @param y,y_hat real values and fitted values.
#' @param sample_weight numeric vector of sample weights, default \code{NULL}.
#' @param correction weight applied to classes with no true samples, default 0.
#' @return geometric mean of per-class recall, a numeric scalar.
#' @export
geometric_mean_of_recall <- function(
    y,
    y_hat,
    sample_weight = NULL,
    correction = 0
) {
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

  # Per-Class Recall: classes with no true samples get recall 0,
  # then all zero recalls are replaced by `correction`
  row_sum                 <- rowSums(cm)
  per_class_recall        <- rep(0, n_labels)
  valid                   <- row_sum > 0
  per_class_recall[valid] <- diag(cm)[valid] / row_sum[valid]
  if (any(!valid)) {
    warning("Recall is ill-defined and set to 0.0 in labels with no true samples.")
  }
  per_class_recall[per_class_recall == 0] <- correction

  # Get G-mean of Recall.
  exp(mean(log(per_class_recall)))
}

