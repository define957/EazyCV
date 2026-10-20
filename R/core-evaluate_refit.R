#' Evaluate a Refitted Model
#'
#' Refit the selected model(s) (unless a refit is supplied), apply the fitted
#' preprocessing transformers to new data, and compute each metric.
#'
#' @param eazycv_model An object of class \code{eazycv_model} returned by
#'   \code{\link{grid_search_cv}()}.
#' @param X_new,y_new New predictor matrix and response. Both are coerced
#'   with \code{as.matrix()} and must have the same number of rows.
#' @param refit Optional result of \code{\link{refit_model}()}. When supplied,
#'   it is used as-is and the seed arguments are ignored.
#' @param refit_seed,corrupt_func_X_seed,corrupt_func_y_seed Passed to
#'   \code{\link{refit_model}()} when \code{refit} is \code{NULL}; see there.
#'
#' @return A data frame with columns \code{model}, \code{metric}, and
#'   \code{value}, one row per metric. \code{model} names the metric whose
#'   best parameters were used.
#'
#' @details
#' Calling this function repeatedly without \code{refit} refits the model each
#' time. Pass a precomputed \code{refit} to evaluate several datasets against
#' the same fitted model, or pass \code{refit_seed} for reproducibility.
#'
#' Metrics are computed on the response scale produced by \code{pipeline_y};
#' if no response pipeline is used, that is the original scale.
#'
#' @seealso \code{\link{refit_model}}, \code{\link{grid_search_cv}}
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
#' evaluate_refit(gs, mtcars[, c("wt", "disp")], mtcars$mpg, refit = refit)
#' }
#' @export
evaluate_refit <- function(
    eazycv_model,
    X_new,
    y_new,
    refit = NULL,
    refit_seed = NULL,
    corrupt_func_X_seed = NULL,
    corrupt_func_y_seed = NULL
) {
  if (!inherits(eazycv_model, "eazycv_model")) {
    stop("'eazycv_model' must be an object returned by grid_search_cv().")
  }

  X_new <- as.matrix(X_new)
  y_new <- as.matrix(y_new)

  if (nrow(X_new) != nrow(y_new)) {
    stop("'X_new' and 'y_new' must have the same number of rows.")
  }

  if (is.null(refit)) {
    refit <- refit_model(
      eazycv_model,
      refit_seed = refit_seed,
      corrupt_func_X_seed = corrupt_func_X_seed,
      corrupt_func_y_seed = corrupt_func_y_seed
    )
  }

  for (t in refit$X_scalers) {
    X_new <- trans(t, X_new)
  }
  for (t in refit$y_scalers) {
    y_new <- trans(t, y_new)
  }

  rows <- list()
  for (m in names(eazycv_model$best_params)) {
    fit <- refit$model_list[[m]]

    if (is.null(fit)) {
      rows[[m]] <- data.frame(
        model = m, metric = m, value = NA_real_,
        stringsAsFactors = FALSE
      )
      next
    }

    predict_res <- predict_model(
      fit,
      X_new,
      y_new,
      eazycv_model$predict_params,
      eazycv_model$predict_func
    )
    metric_res <- metric_evaluate(
      eazycv_model$metrics[[m]],
      y_new,
      predict_res$y_test_hat,
      eazycv_model$metrics_params[[m]]
    )

    rows[[m]] <- data.frame(
      model  = m,
      metric = m,
      value  = metric_res,
      stringsAsFactors = FALSE
    )
  }
  evaluation <- do.call(rbind, rows)
  return(evaluation)
}
