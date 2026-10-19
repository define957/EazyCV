#' Grid Search with K-Fold Cross-Validation
#'
#' Evaluate a grid of model hyperparameters by cross-validation and select
#' the best combination per metric. Refitting is left to the caller via the
#' returned \code{best_params}.
#'
#' @param model A model fitting function; see
#'   \code{K_fold_cross_validation()}.
#' @param X,y Predictor matrix and response; see
#'   \code{K_fold_cross_validation()}.
#' @param folds A list of integer vectors, one per fold, as produced by
#'   \code{make_k_folds()}.
#' @param metrics A (named) list of metric functions. Candidates with any
#'   failed fold are dropped from selection (their aggregated scores are
#'   \code{NA}).
#' @param param_list Named list of parameter vectors, expanded with
#'   \code{expand.grid()} into the search grid. Each grid row is appended
#'   onto \code{model_settings} for one candidate (candidate values override
#'   base settings on name clashes).
#' @param predict_func Prediction function; see
#'   \code{K_fold_cross_validation()}.
#' @param select_direction "min" or "max"; optimization direction, applied to
#'   every metric in \code{metrics}. Keep same-direction metrics together in
#'   one search (e.g. error metrics, or accuracy-like scores).
#' @param pipeline_X,pipeline_y,metrics_params,predict_params,model_settings
#'   See \code{K_fold_cross_validation()}.
#' @param model_seed Optional integer seed. Passed to every candidate's inner
#'   \code{K_fold_cross_validation()} call, so that all candidates are
#'   evaluated under the same random state and the search is reproducible
#'   without touching the global RNG.
#' @param corrupt_func_X,corrupt_func_y Optional contamination functions
#'   applied to the training portion of each fold during candidate
#'   evaluation; see \code{K_fold_cross_validation()}.
#' @param threads_num Number of worker threads used to evaluate candidates
#'   in parallel via \code{foreach}/\code{doSNOW}. Defaults to
#'   \code{parallel::detectCores() - 1}. A PSOCK cluster of that many
#'   workers is spawned for the search and shut down afterwards; a progress
#'   bar is shown while candidates are evaluated. Custom
#'   \code{model}/\code{metrics}/\code{predict_func} closures and the
#'   dataset are exported to the workers automatically.
#' @return An object of class \code{cv_model}, a list with components:
#'   \code{results} (per-candidate metric means, sds, and parameters),
#'   \code{idx_max}/\code{idx_min} (best candidate index per metric, both
#'   directions), \code{best_index} (per \code{select_direction}) and
#'   \code{best_params} (the selected parameter list, one entry per metric,
#'   named by metric), plus \code{num_params}, \code{num_failed}, \code{K},
#'   and \code{time} (total search time). Errors if every candidate failed
#'   for some metric.
#' @seealso \code{\link{K_fold_cross_validation}}, \code{\link{make_k_folds}}
#'
#' @examples
#' \dontrun{
#' rmse <- function(y, y_hat) sqrt(mean((y - y_hat)^2))
#' my_lm <- function(X, y, shift = 0) lm.fit(cbind(1, X), y + shift)
#' my_predict <- function(fit, X) cbind(1, X) %*% fit$coefficients
#' gs <- grid_search_cv(
#'   model = my_lm,
#'   X = mtcars[, c("wt", "disp")],
#'   y = mtcars$mpg,
#'   folds = make_k_folds(nrow(mtcars), 5),
#'   metrics = list(rmse = rmse),
#'   param_list = list(shift = c(0, 0.5, 1)),
#'   predict_func = my_predict,
#'   select_direction = "min"
#' )
#' gs$best_params
#' }
#'
#' @importFrom stats setNames
#' @import foreach
#' @import doSNOW
#' @export
grid_search_cv <- function(
    model,
    X,
    y,
    folds,
    metrics,
    param_list,
    predict_func = predict,
    select_direction = c("min", "max"),
    pipeline_X = NULL,
    pipeline_y = NULL,
    metrics_params = NULL,
    predict_params = NULL,
    model_settings = NULL,
    model_seed = NULL,
    corrupt_func_X = NULL,
    corrupt_func_X_seeds = NULL,
    corrupt_func_y = NULL,
    corrupt_func_y_seeds = NULL,
    threads_num    = parallel::detectCores() - 1
) {

  start_time <- Sys.time()
  withr::local_preserve_seed()
  if (!is.null(model_seed)) set.seed(model_seed)

  param_grid  <- expand.grid(param_list, stringsAsFactors = FALSE)
  n_param     <- nrow(param_grid)
  param_names <- colnames(param_grid)

  select_direction <- match.arg(select_direction)
  metrics          <- metrics_check_cv(metrics)
  num_metric       <- length(metrics)
  metrics_params   <- metrics_params_check_cv(num_metric, metrics_params)
  mnames           <- names(metrics)

  if (is.null(model_settings)) model_settings <- list()

  # search_res <- matrix(NA_real_, n_param, 2 * num_metric)

  cl <- parallel::makeCluster(threads_num)
  on.exit(parallel::stopCluster(cl), add = TRUE)
  op <- options(cli.progress_show_after = 0)
  on.exit(options(op), add = TRUE)
  pb <- cli::cli_progress_bar("Grid search", total = n_param, clear = FALSE)
  on.exit(cli::cli_progress_done(id = pb), add = TRUE)
  progress <- function(n){cli::cli_progress_update(id = pb, set = n)}
  opts <- list(progress = progress)
  doSNOW::registerDoSNOW(cl)
  i <- NULL
  search_res <- foreach::foreach(i = 1:n_param, .combine = rbind,
                             .options.snow = opts) %dopar% {
    params_to_add <- setNames(
      lapply(param_names, function(nm) param_grid[[nm]][[i]]),
      param_names
    )
    res_i <- K_fold_cross_validation(
      model = model,
      X = X,
      y = y,
      folds = folds,
      metrics = metrics,
      predict_func = predict_func,
      pipeline_X = pipeline_X,
      pipeline_y = pipeline_y,
      metrics_params = metrics_params,
      predict_params = predict_params,
      model_settings = c(model_settings, params_to_add),
      model_seed = model_seed,
      corrupt_func_X = corrupt_func_X,
      corrupt_func_X_seeds = corrupt_func_X_seeds,
      corrupt_func_y = corrupt_func_y,
      corrupt_func_y_seeds = corrupt_func_y_seeds
    )
    c(rowMeans(res_i), apply(res_i, 1, sd))
  }

  colnames(search_res) <- c(
    paste0("mean.", mnames),
    paste0("sd.", mnames)
  )
  results <- cbind(as.data.frame(search_res), param_grid)

  idx_max <- stats::setNames(
    vapply(seq_len(num_metric), function(j)
      select_best_idx(search_res[, j], search_res[, num_metric + j], "max"), integer(1)),
    mnames
  )
  idx_min <- stats::setNames(
    vapply(seq_len(num_metric), function(j)
      select_best_idx(search_res[, j], search_res[, num_metric + j], "min"), integer(1)),
    mnames
  )

  best_index  <- if (select_direction == "max") { idx_max } else { idx_min }
  failed_m <- mnames[is.na(best_index)]
  if (length(failed_m) > 0) {
    stop(sprintf("All candidates failed for metric(s): %s; no best parameters.",
                 paste(failed_m, collapse = ", ")))
  }
  best_params <- lapply(best_index, function(b) {
    bp <- as.list(param_grid[b, , drop = FALSE])
    attr(bp, "out.attrs") <- NULL
    c(model_settings, bp)
    }
  )

  structure(
    list(
      results     = results,
      idx_max     = idx_max,
      idx_min     = idx_min,
      best_index  = best_index,
      best_params = best_params,
      num_params  = n_param,
      num_failed  = sum(apply(is.na(search_res[, seq_len(num_metric), drop = FALSE]), 1, all)),
      K           = length(folds),
      time        = Sys.time() - start_time
    ),
    class = "cv_model"
  )
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

# refit_best_model <- function(model, X, y, metric,
#                              idx_max, idx_min, select_direction,
#                              param_grid, model_settings,
#                              pipeline_X = NULL, pipeline_y = NULL,
#                              corrupt_func_X = NULL, corrupt_func_y = NULL) {
#   best_index <- if (select_direction == "max") idx_max[[metric]] else idx_min[[metric]]
#   if (is.na(best_index)) {
#     stop(sprintf("All candidates failed for metric '%s'; cannot refit.", metric))
#   }
#   best_params <- as.list(param_grid[best_index, , drop = FALSE])
#   attr(best_params, "out.attrs") <- NULL
#   X_fit <- as.matrix(X)
#   y_fit <- as.matrix(y)
#   if (!is.null(corrupt_func_X)) {
#     X_fit <- corrupt_func_X(X_fit, 0)
#   }
#   if (!is.null(corrupt_func_y)) {
#     y_fit <- corrupt_func_y(y_fit, 0)
#   }
#   transformers_X <- NULL
#   if (!is.null(pipeline_X)) {
#     transformers_X <- lapply(pipeline_X, function(p) p(X_fit))
#     for (t in transformers_X) {
#       X_fit <- trans(t, X_fit)
#     }
#   }
#   transformers_y <- NULL
#   if (!is.null(pipeline_y)) {
#     transformers_y <- lapply(pipeline_y, function(p) p(y_fit))
#     for (t in transformers_y) {
#       y_fit <- trans(t, y_fit)
#     }
#   }
#   final_fit <- do.call(
#     model,
#     append(
#       list(X = X_fit, y = y_fit),
#       modifyList(if (is.null(model_settings)) list() else model_settings,
#                  best_params)
#     )
#   )
#   list(best_index = best_index, best_params = best_params, final_fit = final_fit,
#        transformers_X = transformers_X, transformers_y = transformers_y)
# }

#' Print Grid-Search Results
#'
#' Prints a header with run information, a table of the best parameters
#' per metric (both max and min directions), and a summary matrix of the
#' best scores (mean and sd).
#'
#' @param x A \code{cv_model} object.
#' @param digits Number of digits for the summary scores. Defaults to 4.
#' @param ... Unused.
#' @importFrom utils capture.output
#' @export
print.cv_model <- function(x, digits = 4, ...) {
  num_metrics  <- length(x$idx_max)
  metric_names <- names(x$idx_max)
  mean_cols    <- paste0("mean.", metric_names)
  sd_cols      <- paste0("sd.", metric_names)
  param_names  <- setdiff(colnames(x$results), c(mean_cols, sd_cols))

  cli::cli_h1("Grid Search and Cross Validation")

  info_names <- c("Number of folds", "Total parameters", "Failed parameters")
  info_vals  <- c(as.character(x$K), as.character(x$num_params),
                  as.character(x$num_failed))
  if (!is.null(x$time)) {
    info_names <- c(info_names, "Time cost")
    info_vals  <- c(info_vals, paste(round(as.numeric(x$time, units = "secs"), 2), "secs"))
  }
  pad <- max(nchar(info_names))
  cli::cli_verbatim(sprintf(paste0("%-", pad, "s: %s"), info_names, info_vals))

  cli::cli_h2("Best Parameters")
  n_col      <- length(param_names) + 1
  col_names  <- c(param_names, "score")
  body       <- matrix("NA", nrow = 2 * num_metrics, ncol = n_col)
  row_labels <- as.character(t(outer(metric_names, c("max", "min"), paste, sep = ".")))
  fmt_score  <- function(v) if (is.na(v)) "NA" else sprintf(paste0("%.", digits, "g"), v)
  for (j in seq_len(num_metrics)) {
    for (k in 1:2) {
      idx <- if (k == 1) x$idx_max[[j]] else x$idx_min[[j]]
      if (is.na(idx)) next
      body[2 * j - 2 + k, ] <- c(
        vapply(x$results[idx, param_names, drop = FALSE], format, character(1)),
        fmt_score(x$results[idx, mean_cols[j]])
      )
    }
  }
  row_width <- max(nchar(row_labels))
  col_width <- pmax(nchar(col_names), apply(nchar(body), 2, max))
  fmt_row <- function(label, values) {
    paste0(sprintf(paste0("%-", row_width, "s "), label),
           paste(sprintf(paste0("%", col_width, "s"), values), collapse = " "))
  }
  header_line <- fmt_row("", col_names)
  div_line <- strrep("\u2500", nchar(header_line))
  lines <- c(header_line, div_line)
  for (j in seq_len(num_metrics)) {
    lines <- c(lines,
               fmt_row(row_labels[2 * j - 1], body[2 * j - 1, ]),
               fmt_row(row_labels[2 * j], body[2 * j, ]),
               div_line)
  }
  cli::cli_verbatim(lines)

  cli::cli_h2("Summary of Metrics")
  score_mat <- matrix(NA_real_, 2, 2 * num_metrics)
  rownames(score_mat) <- c("max", "min")
  colnames(score_mat) <- c(metric_names, paste(metric_names, "- sd"))
  for (j in seq_len(num_metrics)) {
    if (is.na(x$idx_max[[j]])) next
    score_mat[1, j] <- x$results[x$idx_max[[j]], mean_cols[j]]
    score_mat[2, j] <- x$results[x$idx_min[[j]], mean_cols[j]]
    score_mat[1, num_metrics + j] <- x$results[x$idx_max[[j]], sd_cols[j]]
    score_mat[2, num_metrics + j] <- x$results[x$idx_min[[j]], sd_cols[j]]
  }
  mat_lines <- capture.output(
    print(format(round(score_mat, digits), nsmall = digits), quote = FALSE, right = TRUE)
  )
  cli::cli_verbatim(mat_lines)
  cat("\n")
  invisible(x)
}
