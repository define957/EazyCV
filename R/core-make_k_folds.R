#' Create K-Fold Index Splits
#'
#' Generate a list of per-fold test-set row indices, suitable for the
#' \code{folds} argument of \code{K_fold_cross_validation()}.
#'
#' @param num_data Integer >= 1. Total number of samples.
#' @param K Integer >= 1. Number of folds.
#' @param shuffle Logical. If \code{TRUE}, shuffle the sample order before
#'   splitting. Use \code{seed} for reproducibility.
#' @param stratified Logical. If \code{TRUE}, preserve class proportions
#'   within each fold.
#' @param stratify_by Optional vector of class labels of length
#'   \code{num_data}; required when \code{stratified = TRUE}.
#' @param seed Optional integer seed used for the shuffle.
#'
#' @return A list of \code{K} integer vectors. Element \code{i} contains the
#'   row indices of fold \code{i}'s test set.
#'
#' @details
#' With \code{shuffle = FALSE} and \code{stratified = FALSE}, the partition
#' matches \code{sklearn.model_selection.KFold(shuffle=False)}. With
#' \code{stratified = TRUE}, it matches
#' \code{sklearn.model_selection.StratifiedKFold(shuffle=False)}.
#'
#' The random state is preserved: when \code{seed} is given, the global RNG
#' is temporarily set for the shuffle and restored on exit.
#'
#' @examples
#' folds <- make_k_folds(num_data = nrow(mtcars), K = 5)
#' str(folds)
#'
#' # stratified folds keep class proportions
#' y <- ifelse(mtcars$mpg > 20, "high", "low")
#' folds <- make_k_folds(nrow(mtcars), K = 4, stratified = TRUE,
#'                       stratify_by = y)
#' str(folds)
#'
#' @seealso \code{\link{K_fold_cross_validation}},
#'   \code{\link{grid_search_cv}}
#' @author Zhang Jiaqi
#' @export
make_k_folds <- function(
    num_data,
    K = 5,
    shuffle = FALSE,
    stratified = FALSE,
    stratify_by = NULL,
    seed = NULL
  ) {
  if (!is.numeric(num_data) || length(num_data) != 1 || is.na(num_data) ||
      num_data %% 1 != 0 || num_data < 1) {
    stop("'num_data' must be a positive integer.")
  }
  if (!is.numeric(K) || length(K) != 1 || is.na(K) || K %% 1 !=
      0 || K < 1) {
    stop("'K' must be a positive integer.")
  }
  if (K > num_data) {
    stop(sprintf("Cannot have K = %d with num_data = %d.",
                 K, num_data))
  }
  if (stratified == TRUE && is.null(stratify_by)) {
    stop("`stratify_by` must be provided when `stratified = TRUE`.")
  }
  if (!is.null(stratify_by) && any(is.na(stratify_by))) {
    stop("`stratify_by` contains NA.")
  }
  if (!is.null(stratify_by)) {
    stratify_by <- as.vector(stratify_by)
    if (num_data != length(stratify_by)) {
      stop("length of 'stratify_by' does not match 'num_data'.")
    }
  }
  if (!is.null(seed)) {
    withr::local_preserve_seed()
    set.seed(seed)
  }
  perm <- if (shuffle) {
    sample.int(num_data)
  }
  else {
    seq_len(num_data)
  }
  index <- integer(num_data)
  if (stratified) {
    for (cls in unique(stratify_by)) {
      idx_cls <- perm[stratify_by[perm] == cls]
      n_cls <- length(idx_cls)
      if (n_cls < K) {
        stop(sprintf("K = %d cannot be greater than the number of members in each class.", K))
      }
      index[idx_cls] <- sort(rep(1:K, length.out = length(idx_cls)))
    }
  }
  else {
    index[perm] <- sort(rep(1:K, length.out = num_data))
  }
  folds <- lapply(1:K, function(i) {
    which(index == i)
  })
  return(folds)
}
