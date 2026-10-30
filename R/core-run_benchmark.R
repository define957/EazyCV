#' Benchmark Multiple Models with Nested Cross-Validation
#'
#' Run the full nested cross-validation tuning workflow (see
#' \code{\link{nested_cv}()}) once per registered model, on the same outer
#' folds and with the same seeding, and collect per-model score summaries.
#' Because every model sees identical outer splits and inner seeds, the
#' resulting matrices are directly comparable across models.
#'
#' @param registered_models A named list of model specifications, one per
#'   model. Each element is a list with components:
#'   \describe{
#'     \item{\code{fun}}{The model fitting function, or its name as a
#'       character string; passed as \code{model} to \code{nested_cv()}.}
#'     \item{\code{params}}{Named list of candidate hyperparameter values,
#'       expanded with \code{expand.grid()} by the inner grid search; see
#'       \code{param_list} in \code{nested_cv()}. Must be non-empty: models
#'       without tuning parameters should be evaluated with
#'       \code{nested_cv()}/\code{K_fold_cross_validation()} directly.}
#'     \item{\code{setting_keys}}{Optional named list of fixed arguments
#'       passed to \code{model} alongside the tuned ones; see
#'       \code{model_settings} in \code{nested_cv()}.}
#'   }
#' @param X,y Predictor matrix and response, shared by all models; see
#'   \code{\link{nested_cv}()}.
#' @param outer_folds Outer folds, shared by all models; see
#'   \code{\link{nested_cv}()}.
#' @param metrics A metric function or named list of metric functions,
#'   shared by all models; see \code{\link{nested_cv}()}.
#' @param predict_func Prediction function, shared by all models; see
#'   \code{\link{nested_cv}()}.
#' @param select_direction \code{"min"} or \code{"max"}; shared by all
#'   models, see \code{\link{nested_cv}()}.
#' @param pipeline_X,pipeline_y Optional preprocessing pipelines, shared by
#'   all models; see \code{\link{nested_cv}()}.
#' @param metrics_params,predict_params Optional extra argument lists for
#'   \code{metrics} and \code{predict_func}, shared by all models; see
#'   \code{\link{nested_cv}()}.
#' @param model_seeds_list \code{NULL} or a list of length
#'   \code{length(registered_models)}; element \code{j} is passed as
#'   \code{model_seeds} to the run of model \code{j}. See
#'   \code{\link{nested_cv}()}.
#' @param corrupt_func_X,corrupt_func_X_seeds_list,corrupt_func_y,corrupt_func_y_seeds_list
#'   Shared contamination arguments; see \code{\link{nested_cv}()}.
#' @param inner_folds_num,shuffle_inner,stratified_inner,inner_folds_seeds,refit_seeds,refit_corrupt_func_X_seed_vec,refit_corrupt_func_y_seed_vec,threads_num,verbose
#'   Shared tuning and reporting arguments; see \code{\link{nested_cv}()}.
#'   Note that the default \code{inner_folds_num} is 10 here (5 in
#'   \code{nested_cv()}), for stabler benchmark estimates.
#'
#' @return Invisibly, an object of class \code{nested_cv_benchmark}: a list
#'   with two matrices, \code{scores_avg_models} and \code{scores_sd_models},
#'   one row per model (row names from \code{registered_models}) and one
#'   column per metric (column names from \code{metrics}). Cell \code{[j, m]}
#'   is the mean (or standard deviation) over outer folds of metric \code{m}
#'   for model \code{j}. Aggregation uses \code{na.rm = FALSE}: a single
#'   failed outer fold turns the whole cell into \code{NA}.
#'
#' @details
#' The workload per model is
#' \code{length(outer_folds) * nrow(expand.grid(params)) * inner_folds_num}
#' fits during the search plus one refit per outer fold and metric; the
#' verbose output prints this count for each model before fitting starts.
#'
#' Models are fitted sequentially; within each model the inner grid search
#' parallelizes as described in \code{\link{nested_cv}()}.
#'
#' @seealso \code{\link{nested_cv}}, \code{\link{grid_search_cv}},
#'   \code{\link{make_k_folds}}
#'
#' @examples
#' \dontrun{
#' rmse <- function(y, y_hat) sqrt(mean((y - y_hat)^2))
#' my_lm <- function(X, y, shift = 0) lm.fit(cbind(1, X), y + shift)
#' my_predict <- function(fit, X) cbind(1, X) %*% fit$coefficients
#'
#' X <- mtcars[, c("wt", "disp")]
#' y <- mtcars$mpg
#'
#' outer <- make_k_folds(nrow(mtcars), K = 3, shuffle = TRUE, seed = 1)
#'
#' registered <- list(
#'   shift_small = list(fun = my_lm, params = list(shift = c(0, 0.5))),
#'   shift_wide  = list(fun = my_lm, params = list(shift = c(0, 0.5, 1, 2)))
#' )
#'
#' res <- run_benchmark(
#'   registered_models   = registered,
#'   X                   = X,
#'   y                   = y,
#'   outer_folds         = outer,
#'   metrics             = list(rmse = rmse),
#'   predict_func        = my_predict,
#'   select_direction    = "min",
#'   inner_folds_num     = 3,
#'   inner_folds_seeds   = c(11, 12, 13),
#'   model_seeds_list    = list(c(101, 102, 103), c(104, 105, 106)),
#'   refit_seeds         = c(201, 202, 203),
#'   threads_num         = 2
#' )
#'
#' res$scores_avg_models  # one row per model
#' }
#'
#' @author Zhang Jiaqi
#' @export
run_benchmark <- function(
    registered_models,
    X,
    y,
    outer_folds,
    metrics,
    predict_func                  = predict,
    select_direction              = c("min", "max"),
    pipeline_X                    = NULL,
    pipeline_y                    = NULL,
    metrics_params                = NULL,
    predict_params                = NULL,
    model_seeds_list              = NULL,
    corrupt_func_X                = NULL,
    corrupt_func_X_seeds_list     = NULL,
    corrupt_func_y                = NULL,
    corrupt_func_y_seeds_list     = NULL,
    inner_folds_num               = 10,
    shuffle_inner                 = TRUE,
    stratified_inner              = FALSE,
    inner_folds_seeds             = NULL,
    refit_seeds                   = NULL,
    refit_corrupt_func_X_seed_vec = NULL,
    refit_corrupt_func_y_seed_vec = NULL,
    threads_num                   = parallel::detectCores() - 1,
    verbose                       = TRUE
) {

  num_models  <- length(registered_models)
  num_metrics <- length(metrics)

  scores_avg_models <- matrix(NA_real_, num_models, num_metrics)
  scores_sd_models  <- matrix(NA_real_, num_models, num_metrics)
  for (j in seq_along(registered_models)) {
    spec <- registered_models[[j]]
    model_i  <- if (is.character(spec$fun)) { match.fun(spec$fun) } else { spec$fun }
    t0 <- Sys.time()
    if (verbose) {
      n_grid <- prod(vapply(spec$params, length, integer(1)))
      n_fits <- n_grid * length(outer_folds) * inner_folds_num
      model_name <- names(registered_models)[j]
      cli::cli_h3("Model {j}/{num_models}: {model_name}")
      cli::cli_text(
        "{.dim grid {n_grid} candidates | {length(outer_folds)} outer x {inner_folds_num} inner folds = {n_fits} fits}")
    }
    scores <- nested_cv(
      model_i,
      X,
      y,
      outer_folds,
      metrics,
      param_list                    = spec$params,
      predict_func                  = predict_func,
      select_direction              = select_direction,
      pipeline_X                    = pipeline_X,
      pipeline_y                    = pipeline_y,
      metrics_params                = metrics_params,
      predict_params                = predict_params,
      model_settings                = spec$setting_keys,
      model_seeds                   = model_seeds_list[[j]],
      corrupt_func_X                = corrupt_func_X,
      corrupt_func_X_seeds_list     = corrupt_func_X_seeds_list,
      corrupt_func_y                = corrupt_func_y,
      corrupt_func_y_seeds_list     = corrupt_func_y_seeds_list,
      inner_folds_num               = inner_folds_num,
      shuffle_inner                 = shuffle_inner,
      stratified_inner              = stratified_inner,
      inner_folds_seeds             = inner_folds_seeds,
      refit_seeds                   = refit_seeds,
      refit_corrupt_func_X_seed_vec = refit_corrupt_func_X_seed_vec,
      refit_corrupt_func_y_seed_vec = refit_corrupt_func_y_seed_vec,
      threads_num                   = threads_num,
      verbose                       = verbose
    )
    scores_avg_models[j, ] <- apply(scores, 2, mean)
    scores_sd_models[j, ]  <- apply(scores, 2, sd)
    if (verbose) {
      elapsed_sec <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
      if (elapsed_sec < 60) {
        elapsed_txt <- sprintf("%.1f s", elapsed_sec)
      } else if (elapsed_sec < 3600) {
        elapsed_txt <- sprintf("%.1f min", elapsed_sec / 60)
      } else {
        elapsed_txt <- sprintf("%.2f h", elapsed_sec / 3600)
      }
      model_name <- names(registered_models)[j]
      cli::cli_alert_success("{model_name} finished in {elapsed_txt}")
    }
  }
  rownames(scores_avg_models) <- names(registered_models)
  colnames(scores_avg_models) <- names(metrics)
  rownames(scores_sd_models)  <- names(registered_models)
  colnames(scores_sd_models)  <- names(metrics)
  res <- list(scores_avg_models = scores_avg_models,
              scores_sd_models  = scores_sd_models)
  class(res) <- "nested_cv_benchmark"
  if (verbose) print(res)
  invisible(res)
}

#' Print Nested CV Benchmark Results
#'
#' Prints a header with run information and a summary matrix of
#' \code{mean} \eqn{\pm} \code{sd} per model and metric, in the style of
#' EazyCV print methods.
#'
#' @param x A \code{nested_cv_benchmark} object.
#' @param digits Number of digits to display. Defaults to 4.
#' @param ... Unused.
#' @author Zhang Jiaqi
#' @export
print.nested_cv_benchmark <- function(x, digits = 4, ...) {
  fmt <- function(v) ifelse(is.na(v), "NA",
                            format(round(v, digits), nsmall = digits))

  avg_mat <- matrix(fmt(x$scores_avg_models), nrow = nrow(x$scores_avg_models))
  sd_mat  <- matrix(fmt(x$scores_sd_models),  nrow = nrow(x$scores_sd_models))

  # align the "\u00b1" within each column by right-justifying the mean part
  cell_mat <- vapply(seq_len(ncol(avg_mat)), function(j) {
    mean_w <- max(nchar(avg_mat[, j]))
    paste(format(avg_mat[, j], width = mean_w, justify = "right"),
          "\u00b1", sd_mat[, j])
  }, character(nrow(avg_mat)))
  dim(cell_mat) <- c(nrow(avg_mat), ncol(avg_mat))
  dimnames(cell_mat) <- dimnames(x$scores_avg_models)

  cli::cli_h1("Nested CV Benchmark")
  info_names <- c("Number of models", "Number of metrics")
  info_vals  <- c(as.character(nrow(cell_mat)), as.character(ncol(cell_mat)))
  pad <- max(nchar(info_names))
  cli::cli_verbatim(sprintf(paste0("%-", pad, "s: %s"), info_names, info_vals))

  cli::cli_h2("Summary of Metrics (mean \u00b1 sd)")
  col_width <- pmax(nchar(colnames(cell_mat)),
                    apply(cell_mat, 2, function(col) max(nchar(col))))
  row_width <- max(nchar(rownames(cell_mat)))
  header <- paste0(sprintf(paste0("%", row_width, "s "), ""),
                   paste(sprintf(paste0(" %", col_width, "s "), colnames(cell_mat)),
                         collapse = "|"))
  sep_line  <- strrep("-", nchar(header))
  body <- paste0(
    sprintf(paste0("%-", row_width, "s "), rownames(cell_mat)),
    apply(cell_mat, 1, function(r)
      paste0(" ", paste(sprintf(paste0("%", col_width, "s"), r),
                        collapse = " | "), " "))
  )
  cli::cli_verbatim(c(header, sep_line, body))
  cat("\n")
  invisible(x)
}

