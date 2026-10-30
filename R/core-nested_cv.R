#' Nested K-Fold Cross-Validation
#'
#' Estimate the performance of a complete tuning procedure by nesting a grid
#' search inside an outer cross-validation loop. For each outer fold the
#' hyperparameters are selected by \code{\link{grid_search_cv}()} on the outer
#' training set only; the selected model is then refitted on that same training
#' set and scored once on the untouched outer test fold. Because the outer test
#' folds never take part in selection or fitting, the returned scores estimate
#' what the whole workflow (tuning included) would achieve on new data, rather
#' than the optimistic CV score of a configuration picked in advance.
#'
#' @param model A model fitting function with signature
#'   \code{function(X, y, ...)}; see \code{K_fold_cross_validation()}.
#' @param X,y Predictor matrix and response. Both are coerced with
#'   \code{as.matrix()} and must have the same number of rows.
#' @param outer_folds A list of integer vectors, one per outer fold, each
#'   holding the row indices that are held out for testing (the same convention
#'   as \code{folds} in \code{K_fold_cross_validation()}). Generate them with
#'   \code{make_k_folds()}, using \code{stratified = TRUE} for classification.
#'   The folds must partition \code{1:nrow(X)}: every row is tested exactly
#'   once, with no overlap and no gaps. The number of outer folds is
#'   \code{length(outer_folds)}.
#' @param metrics A metric function, or a (named) list of metric functions, as
#'   accepted by \code{K_fold_cross_validation()}. One configuration is
#'   selected and refitted per metric, following \code{select_direction}.
#' @param param_list Named list of candidate values, expanded with
#'   \code{expand.grid()} by \code{grid_search_cv()}; see there.
#' @param predict_func Prediction function, see
#'   \code{K_fold_cross_validation()}. Defaults to \code{predict}.
#' @param select_direction \code{"min"} or \code{"max"}; the optimization
#'   direction applied to every metric in the inner search. Metrics of
#'   opposite direction (for example an accuracy and an error) must be run in
#'   separate calls.
#' @param pipeline_X,pipeline_y Optional lists of preprocessing functions.
#'   They are fitted on the outer training set of every outer fold, and on the
#'   inner training folds during the search, so no test observation ever
#'   contributes to a transformer.
#' @param metrics_params,predict_params,model_settings See
#'   \code{K_fold_cross_validation()}.
#' @param model_seeds \code{NULL} or an integer vector of length
#'   \code{length(outer_folds)}. Element \code{i} is the \code{model_seed}
#'   used by the inner search of outer fold \code{i}; inner fold \code{k} is
#'   then seeded with \code{model_seeds[i] + k}. With \code{NULL} the global
#'   RNG is consumed.
#' @param corrupt_func_X,corrupt_func_y Optional contamination functions,
#'   applied to the training portion of a fold before any preprocessing; see
#'   \code{K_fold_cross_validation()}. During the search, inner fold \code{k}
#'   of outer fold \code{i} uses \code{corrupt_func_X_seeds_list[[i]][k]};
#'   the final refit of outer fold \code{i} uses
#'   \code{refit_corrupt_func_X_seed_vec[i]}. Outer test folds are never
#'   contaminated.
#' @param corrupt_func_X_seeds_list,corrupt_func_y_seeds_list Required when the
#'   corresponding contamination function is supplied. A list of length
#'   \code{length(outer_folds)} whose element \code{i} is an integer vector of
#'   length \code{inner_folds_num}, giving the contamination seed of each
#'   inner fold within outer fold \code{i}.
#' @param inner_folds_num Integer >= 2. Number of inner folds used to select
#'   the hyperparameters. Defaults to 5.
#' @param shuffle_inner,stratified_inner Passed to \code{make_k_folds()} for
#'   the inner split. \code{stratified_inner} stratifies by the outer training
#'   response and is only meaningful for classification; keep it \code{FALSE}
#'   for a continuous response.
#' @param inner_folds_seeds \code{NULL} or an integer vector of length
#'   \code{length(outer_folds)}. Element \code{i} is the \code{seed} given to
#'   \code{make_k_folds()} for the inner split of outer fold \code{i}. With
#'   \code{NULL} the global RNG is consumed and the inner splits are not
#'   reproducible.
#' @param refit_seeds \code{NULL} or an integer vector of length
#'   \code{length(outer_folds)}. Element \code{i} is the \code{refit_seed}
#'   passed to \code{evaluate_refit()} for outer fold \code{i}.
#' @param refit_corrupt_func_X_seed_vec,refit_corrupt_func_y_seed_vec
#'   \code{NULL} or an integer vector of length \code{length(outer_folds)}.
#'   Element \code{i} is the contamination seed used when the model selected
#'   in outer fold \code{i} is refitted. Requires the matching contamination
#'   function.
#' @param threads_num Number of worker threads used by each inner grid search.
#'   Defaults to \code{max(1, parallel::detectCores() - 1)}; values below 1
#'   fall back to 1, and the value is capped at the number of candidates in
#'   \code{param_list}. A PSOCK cluster is created and stopped for every outer
#'   fold; outer folds are processed sequentially.
#' @param verbose Logical; whether to print the one-line summary of each outer
#'   fold. Defaults to \code{TRUE}.
#'
#' @return Invisibly, a numeric matrix with one row per outer fold and one
#'   column per metric, columns named after the metrics (unnamed metric lists
#'   are named \code{metric-1}, \code{metric-2}, \ldots). Cell
#'   \code{[i, j]} is metric \code{j} computed once on outer test fold
#'   \code{i}, using the model selected for metric \code{j} from the outer
#'   training data of that fold alone. A cell is \code{NA} when the fold could
#'   not be evaluated.
#'
#' @details
#' For outer fold \code{i} the function performs three steps that never see
#' the outer test fold:
#' \enumerate{
#'   \item split the outer training set into \code{inner_folds_num} inner
#'     folds with \code{make_k_folds()};
#'   \item run the full grid search with \code{grid_search_cv()} on those
#'     inner folds, selecting one configuration per metric;
#'   \item refit the selected configuration on the outer training set and
#'     compute every metric on the outer test fold.
#' }
#'
#' Because steps 1--2 only use the outer training rows, the score in step 3 is
#' not inflated by the selection. The price is variance: each outer training
#' set is smaller than the full data, and the inner search adds its own
#' randomness.
#'
#' Selection is per metric, so different columns of the result may come from
#' different configurations and different fitted models. If a single model is
#' wanted, select on one metric and report the others alongside.
#'
#' \strong{Reproducibility.} A run is fully reproducible when the outer split
#' and all seed vectors are supplied: \code{outer_folds} from
#' \code{make_k_folds(seed = )}, plus \code{inner_folds_seeds},
#' \code{model_seeds} and \code{refit_seeds}, each of length
#' \code{length(outer_folds)}. The length and type of these vectors are
#' validated before any fitting starts. When every seed is supplied, the
#' global RNG state is left unchanged on exit; leaving any of them
#' \code{NULL} falls back to the global RNG.
#'
#' \strong{Cost.} One call fits
#' \code{length(outer_folds) * nrow(expand.grid(param_list)) * inner_folds_num}
#' models during the search, plus one refit per outer fold and metric. Inner
#' training sets contain roughly \code{nrow(X) * (1 - 1 / inner_folds_num)}
#' rows times \code{(1 - 1 / length(outer_folds))}, which matters for models
#' with a minimum sample size.
#'
#' \strong{Failures.} An inner fold that raises an error contributes \code{NA};
#' \code{grid_search_cv()} drops any candidate with a failed fold. If every
#' candidate fails for a metric, the outer fold stops with an error.
#'
#' @seealso \code{\link{make_k_folds}}, \code{\link{grid_search_cv}},
#'   \code{\link{refit_model}}, \code{\link{evaluate_refit}},
#'   \code{\link{K_fold_cross_validation}}
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
#' scores <- nested_cv(
#'   model             = my_lm,
#'   X                 = X,
#'   y                 = y,
#'   outer_folds       = outer,
#'   metrics           = list(rmse = rmse),
#'   param_list        = list(shift = c(0, 0.5, 1)),
#'   predict_func      = my_predict,
#'   select_direction  = "min",
#'   inner_folds_num   = 3,
#'   inner_folds_seeds = c(11, 12, 13),
#'   model_seeds       = c(101, 102, 103),
#'   refit_seeds       = c(201, 202, 203),
#'   threads_num       = 2
#' )
#'
#' scores            # one row per outer fold
#' colMeans(scores)  # nested-CV estimate of the tuning procedure
#' }
#'
#' @author Zhang Jiaqi
#' @export
nested_cv <- function(
    model,
    X,
    y,
    outer_folds,
    metrics,
    param_list,
    predict_func                  = predict,
    select_direction              = c("min", "max"),
    pipeline_X                    = NULL,
    pipeline_y                    = NULL,
    metrics_params                = NULL,
    predict_params                = NULL,
    model_settings                = NULL,
    model_seeds                   = NULL,
    corrupt_func_X                = NULL,
    corrupt_func_X_seeds_list     = NULL,
    corrupt_func_y                = NULL,
    corrupt_func_y_seeds_list     = NULL,
    inner_folds_num               = 5,
    shuffle_inner                 = TRUE,
    stratified_inner              = FALSE,
    inner_folds_seeds             = NULL,
    refit_seeds                   = NULL,
    refit_corrupt_func_X_seed_vec = NULL,
    refit_corrupt_func_y_seed_vec = NULL,
    threads_num                   = max(parallel::detectCores() - 1L, 1L, na.rm = TRUE),
    verbose                       = TRUE
) {
  start_time <- Sys.time()
  threads_num <- max(1L, suppressWarnings(as.integer(threads_num))[1], na.rm = TRUE)

  X                <- as.matrix(X)
  y                <- as.matrix(y)
  metrics          <- metrics_check_cv(metrics)
  select_direction <- match.arg(select_direction)

  if (nrow(X) != nrow(y)) {
    stop("'X' and 'y' must have the same number of rows.", call. = FALSE)
  }
  if (!is.numeric(inner_folds_num) || length(inner_folds_num) != 1L ||
      is.na(inner_folds_num) || inner_folds_num %% 1 != 0 ||
      inner_folds_num < 2L) {
    stop("'inner_folds_num' must be a single integer >= 2.", call. = FALSE)
  }
  inner_folds_num <- as.integer(inner_folds_num)

  outer_folds_num  <- length(outer_folds)

  nested_cv_seed_vec_check(model_seeds,       outer_folds_num, "model_seeds")
  nested_cv_seed_vec_check(inner_folds_seeds, outer_folds_num, "inner_folds_seeds")
  nested_cv_seed_vec_check(refit_seeds,       outer_folds_num, "refit_seeds")
  nested_cv_seed_vec_check(refit_corrupt_func_X_seed_vec, outer_folds_num,
                           "refit_corrupt_func_X_seed_vec")
  nested_cv_seed_vec_check(refit_corrupt_func_y_seed_vec, outer_folds_num,
                           "refit_corrupt_func_y_seed_vec")

  nested_cv_corrupt_seeds_list_check(corrupt_func_X_seeds_list, outer_folds_num,
                                     inner_folds_num, "corrupt_func_X_seeds_list",
                                     corrupt_func_X)
  nested_cv_corrupt_seeds_list_check(corrupt_func_y_seeds_list, outer_folds_num,
                                     inner_folds_num, "corrupt_func_y_seeds_list",
                                     corrupt_func_y)

  scores <- matrix(
    NA_real_,
    outer_folds_num,
    length(metrics),
    dimnames = list(NULL, names(metrics))
  )
  for (i in seq_along(outer_folds)) {
    corrupt_func_X_seed <- NULL
    corrupt_func_y_seed <- NULL

    idx_test <- outer_folds[[i]]
    X_train   <- X[-idx_test, , drop = FALSE]
    y_train   <- y[-idx_test, drop = FALSE]
    X_test    <- X[idx_test, , drop = FALSE]
    y_test    <- y[idx_test, drop = FALSE]

    inner_folds <- make_k_folds(
      num_data    = nrow(X_train),
      K           = inner_folds_num,
      shuffle     = shuffle_inner,
      stratified  = stratified_inner,
      stratify_by = if (stratified_inner) { y_train } else { NULL },
      seed        = inner_folds_seeds[i]
    )

    cv_args <- list(
      model                = model,
      X                    = X_train,
      y                    = y_train,
      folds                = inner_folds,
      metrics              = metrics,
      param_list           = param_list,
      predict_func         = predict_func,
      select_direction     = select_direction,
      pipeline_X           = pipeline_X,
      pipeline_y           = pipeline_y,
      metrics_params       = metrics_params,
      predict_params       = predict_params,
      model_settings       = model_settings,
      model_seed           = model_seeds[i],
      corrupt_func_X       = corrupt_func_X,
      corrupt_func_X_seeds = corrupt_func_X_seeds_list[[i]],
      corrupt_func_y       = corrupt_func_y,
      corrupt_func_y_seeds = corrupt_func_y_seeds_list[[i]],
      threads_num          = threads_num
    )

    gs  <- do.call(grid_search_cv, cv_args)

    if (!is.null(gs$corrupt_func_X) && !is.null(refit_corrupt_func_X_seed_vec)) {
      corrupt_func_X_seed <- refit_corrupt_func_X_seed_vec[i]
    }
    if (!is.null(gs$corrupt_func_y) && !is.null(refit_corrupt_func_y_seed_vec)) {
      corrupt_func_y_seed <- refit_corrupt_func_y_seed_vec[i]
    }
    if (!is.null(refit_seeds)) {
      refit_seed <- refit_seeds[i]
    } else {
      refit_seed <- NULL
    }
    ev <- evaluate_refit(
      gs,
      X_test,
      y_test,
      refit = NULL,
      refit_seed = refit_seed,
      corrupt_func_X_seed = corrupt_func_X_seed,
      corrupt_func_y_seed = corrupt_func_y_seed
    )
    scores[i, ] <- ev$value[match(names(metrics), ev$metric)]

    inner_mean <- vapply(names(metrics), function(m) {
      b <- gs$best_index[[m]]
      if (is.na(b)) NA_real_ else gs$results[b, paste0("mean.", m)]
    }, numeric(1))

    n_ok <- gs$num_params - gs$num_failed

    if (verbose) {
      m_width <- max(nchar(names(metrics)))
      metric_lines <- vapply(names(metrics), function(m) {
        sprintf("    %-*s  inner: %7.3f   outer: %7.3f",
                m_width, m,
                ifelse(is.na(inner_mean[m]), NA_real_, inner_mean[m]),
                ifelse(is.na(ev$value[match(m, ev$metric)]), NA_real_,
                       ev$value[match(m, ev$metric)]))
      }, character(1))

      cli::cli_h2(sprintf("Outer Fold %d/%d", i, length(outer_folds)))
      cli::cli_verbatim(c(
        sprintf("    Grid search: %d/%d candidates ok (%d failed)",
                n_ok, gs$num_params, gs$num_failed),
        metric_lines
      ))
    }
  }
  end_time <- Sys.time()
  invisible(scores)
}

# ---- internal argument checks -------------------------------------------------

#' @noRd
nested_cv_seed_vec_check <- function(x, n, arg) {
  if (is.null(x)) return(invisible(NULL))
  if (!is.numeric(x))
    stop(sprintf("'%s' must be NULL or a numeric vector, not %s.",
                 arg, paste(class(x), collapse = "/")), call. = FALSE)
  if (length(x) != n)
    stop(sprintf("'%s' must have length %d (one seed per outer fold), not %d.",
                 arg, n, length(x)), call. = FALSE)
  if (anyNA(x) || any(!is.finite(x)) || any(x %% 1 != 0))
    stop(sprintf("'%s' must contain integer seeds only (no NA/NaN/Inf/non-integer).",
                 arg), call. = FALSE)
  invisible(NULL)
}

#' @noRd
nested_cv_corrupt_seeds_list_check <- function(x, n_outer, n_inner, arg, func) {
  if (is.null(x)) return(invisible(NULL))
  if (!is.list(x) || length(x) != n_outer)
    stop(sprintf("'%s' must be a list of length %d (one vector per outer fold).",
                 arg, n_outer), call. = FALSE)
  for (i in seq_len(n_outer))
    corrupt_seeds_check_cv(x[[i]], n_inner, sprintf("%s[[%d]]", arg, i), func)
  invisible(NULL)
}
