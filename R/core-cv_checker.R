metrics_check_cv <- function(metrics) {
  if (is.function(metrics)) {
    metrics <- list(metrics)
  }
  if (!is.list(metrics)) {
    stop("'metrics' must be a metric function or a list of metric functions.")
  }
  nm <- names(metrics)
  if (is.null(nm)) {
    nm <- rep("", length(metrics))
  }
  missing_nm <- !nzchar(nm)
  if (any(missing_nm)) {
    nm[missing_nm] <- paste0("metric-", which(missing_nm))
    names(metrics) <- nm
  }
  return(metrics)
}

metrics_params_check_cv <- function(num_metrics, metrics_params) {
  if (is.null(metrics_params)) {
    metrics_params <- vector("list", num_metrics)
  } else {
    if (!is.list(metrics_params)) {
      stop("'metrics_params' must be a list.")
    }
    len_params <- length(metrics_params)
    if (len_params != num_metrics) {
      stop(sprintf("Length of 'metrics_params' (%d) does not match number of metrics (%d).", len_params, num_metrics))
    }
  }
  return(metrics_params)
}

metric_evaluate <- function(metric_func, y, y_hat, metric_params) {
  metric_params <- append(list(y = y, y_hat = y_hat), metric_params)
  evaluate_res <- do.call(metric_func, metric_params)
  return(evaluate_res)
}

predict_model <- function(model_res, X_test, y_test,
                          predict_params, predict_func) {
  y_test_hat <- do.call(
    predict_func,
    append(list(model_res, X_test), predict_params)
  )
  return(list(y_test_hat = y_test_hat))
}

corrupt_seeds_check_cv <- function(seeds, num_folds, arg, func = NULL) {
  func_arg <- sub("_seeds$", "", arg)
  if (is.null(seeds)) {
    if (!is.null(func)) {
      stop(sprintf("'%s' must be supplied when '%s' is used.", arg, func_arg))
    }
    return(invisible(NULL))
  }
  if (!is.numeric(seeds) || length(seeds) != num_folds || anyNA(seeds) ||
      any(seeds %% 1 != 0)) {
    stop(sprintf(
      "'%s' must be an integer vector of length %d with no missing values.",
      arg, num_folds
    ))
  }
  if (is.null(func)) {
    stop(sprintf("'%s' requires '%s'.", arg, func_arg))
  }
  invisible(NULL)
}
