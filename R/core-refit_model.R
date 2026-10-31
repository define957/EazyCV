#' Refit the Selected Model(s)
#'
#' Fit final model(s) on the full data using the best hyperparameters selected
#' by \code{\link{grid_search_cv}()}. One model is fitted per metric, using
#' that metric's best parameter combination.
#'
#' @param eazycv_model An object of class \code{eazycv_model} returned by
#'   \code{\link{grid_search_cv}()}.
#' @param refit_seed Optional integer seed for the preprocessing pipeline and
#'   model fitting. When supplied, the global RNG state is restored on exit.
#'   Contamination noise is controlled separately by
#'   \code{corrupt_func_X_seed} and \code{corrupt_func_y_seed}.
#' @param corrupt_func_X_seed,corrupt_func_y_seed Optional scalar seeds for the
#'   contamination functions stored in \code{eazycv_model}. Each seed is
#'   applied locally via \code{withr::with_seed()}, so contamination draws do
#'   not consume the \code{refit_seed} stream. Supplying a seed requires the
#'   corresponding contamination function to be present.
#'
#' @return A list with components:
#' \itemize{
#'   \item \code{model_list}: named list of fitted models, one per metric;
#'   \item \code{X_scalers}: transformers fitted on the full predictor data,
#'     in pipeline order;
#'   \item \code{y_scalers}: transformers fitted on the full response data,
#'     in pipeline order.
#' }
#'
#' @details
#' The order of operations matches cross-validation:
#' \code{pipeline_X}/\code{pipeline_y}, then contamination, then model
#' fitting. Pipelines are fitted on the clean full data and contamination is
#' applied afterwards, so noise is defined in the transformed units, exactly
#' as during cross-validation. The returned transformers should be applied to
#' new data before prediction (as \code{\link{evaluate_refit}()} does).
#'
#' With \code{refit_seed = NULL} the global RNG is used and consumed. Passing
#' \code{refit_seed} makes the refit reproducible and leaves the global RNG
#' unchanged. The two contamination seeds are independent of
#' \code{refit_seed}; if they are \code{NULL}, contamination draws come from
#' the \code{refit_seed} stream (or the global RNG when that is also
#' \code{NULL}).
#'
#' @seealso \code{\link{evaluate_refit}}, \code{\link{grid_search_cv}}
#'
#' @examples
#' \dontrun{
#' rmse <- function(y, y_hat) sqrt(mean((y - y_hat)^2))
#' gs <- grid_search_cv(
#'   model = function(X, y, shift = 0) lm.fit(cbind(1, X), y + shift),
#'   X = mtcars[, c("wt", "disp")],
#'   y = mtcars$mpg,
#'   folds = make_k_folds(nrow(mtcars), 5, shuffle = TRUE, seed = 1),
#'   metrics = list(rmse = rmse),
#'   param_list = list(shift = c(0, 0.5, 1)),
#'   predict_func = function(fit, X) cbind(1, X) %*% fit$coefficients,
#'   select_direction = "min",
#'   model_seed = 42
#' )
#' refit <- refit_model(gs, refit_seed = 7)
#' }
#' @export
refit_model <- function(
    eazycv_model,
    refit_seed = NULL,
    corrupt_func_X_seed = NULL,
    corrupt_func_y_seed = NULL) {

  if (!is.null(refit_seed)) {
    withr::local_preserve_seed()
    set.seed(refit_seed)
  }

  if (!is.null(corrupt_func_X_seed) && is.null(eazycv_model$corrupt_func_X)) {
    stop("'corrupt_func_X_seed' requires 'eazycv_model$corrupt_func_X'.")
  }
  if (!is.null(corrupt_func_y_seed) && is.null(eazycv_model$corrupt_func_y)) {
    stop("'corrupt_func_y_seed' requires 'eazycv_model$corrupt_func_y'.")
  }

  model_list <- list()

  X <- as.matrix(eazycv_model$X)
  y <- as.matrix(eazycv_model$y)

  X_scalers <- list()
  y_scalers <- list()
  if (!is.null(eazycv_model$pipeline_X)) {
    for (pipi in seq_along(eazycv_model$pipeline_X)) {
      pip_temp <- eazycv_model$pipeline_X[[pipi]](X)
      X_scalers[[pipi]] <- pip_temp
      X <- trans(pip_temp, X)
    }
  }

  if (!is.null(eazycv_model$pipeline_y)) {
    for (pipi in seq_along(eazycv_model$pipeline_y)) {
      pip_temp <- eazycv_model$pipeline_y[[pipi]](y)
      y_scalers[[pipi]] <- pip_temp
      y  <- trans(pip_temp, y)
    }
  }

  if (!is.null(eazycv_model$corrupt_func_X)) {
    X <- if (is.null(corrupt_func_X_seed)) {
      eazycv_model$corrupt_func_X(X)
    } else {
      withr::with_seed(corrupt_func_X_seed, eazycv_model$corrupt_func_X(X))
    }
  }
  if (!is.null(eazycv_model$corrupt_func_y)) {
    y <- if (is.null(corrupt_func_y_seed)) {
      eazycv_model$corrupt_func_y(y)
    } else {
      withr::with_seed(corrupt_func_y_seed, eazycv_model$corrupt_func_y(y))
    }
  }

  for (m in names(eazycv_model$best_params)) {
    model_list[[m]] <- do.call(eazycv_model$model,
                               c(list(
                                 X = X,
                                 y = y
                               ),
                               eazycv_model$best_params[[m]]
                               )
    )
  }
  list(model_list = model_list, X_scalers = X_scalers, y_scalers = y_scalers)
}

select_best_idx <- function(scores, sds, direction = c("max", "min")) {
  direction <- match.arg(direction)
  if (all(is.na(scores))) { return(NA_integer_) }
  extreme <- if (direction == "max") max(scores, na.rm = TRUE) else min(scores, na.rm = TRUE)
  ties <- which(scores == extreme)
  if (length(ties) > 1) {
    if (!all(is.na(sds[ties]))) {
      ties <- ties[which.min(sds[ties])]
    } else {
      ties <- ties[1]
    }
  }
  return(ties)
}
