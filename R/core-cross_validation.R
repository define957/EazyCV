#' K-Fold Cross-Validation
#'
#' Perform k-fold cross-validation for an arbitrary model. Using the fold
#' splits supplied in \code{folds}, the function trains the model on the
#' training portion of each fold, evaluates predictions on the hold-out
#' portion, and returns a matrix of metric values.
#'
#' @param model A model fitting function with signature
#'   \code{function(X, y, ...)}. \code{X} will be passed as a matrix and
#'   \code{y} as a matrix/vector. Additional arguments are supplied via
#'   \code{model_settings}.
#' @param X Predictor matrix. Will be coerced to \code{matrix} via
#'   \code{as.matrix()}.
#' @param y Response vector or matrix. Will be coerced to \code{matrix} via
#'   \code{as.matrix()}.
#' @param folds A list of integer vectors, one per fold. Each element contains
#'   the row indices of that fold's test set; training indices are taken as
#'   the complement. Generate standard splits with \code{make_k_folds()}.
#' @param metrics A metric function, or a \code{list} of metric functions.
#'   Each metric must accept \code{y} (ground truth) and \code{y_hat}
#'   (predictions) as the first two arguments. If the list is unnamed or
#'   partially named, missing names default to
#'   \code{metric-1}, \code{metric-2}, \ldots.
#' @param predict_func A prediction function with signature
#'   \code{function(model_fit, X_test, ...)}. Defaults to \code{predict}.
#'   Must return a vector of predictions.
#' @param pipeline_X A list of preprocessing functions for \code{X}. Each
#'   function must accept a training matrix and return a transformer object.
#'   The transformer is then applied to both training and test data via
#'   \code{trans()}. Default is \code{NULL} (no preprocessing).
#' @param pipeline_y A list of preprocessing functions for \code{y}. Applied
#'   to the training response before model fitting, and to the test response
#'   before metric evaluation. Default is \code{NULL} (no preprocessing).
#' @param metrics_params A list of parameter lists, one per metric. Passed to
#'   each metric function as additional arguments after \code{y} and
#'   \code{y_hat}. Default is \code{NULL}.
#' @param predict_params Additional arguments passed to \code{predict_func}
#'   as \code{...}. Default is \code{NULL}.
#' @param model_settings Additional arguments passed to \code{model} as
#'   \code{...}. Default is \code{NULL}.
#' @param model_seed Optional integer seed. If provided, the random number
#'   generator state is temporarily set via \code{withr::with_seed()} for the
#'   duration of the cross-validation and restored afterwards, ensuring
#'   reproducibility without side effects on the global RNG state.
#' @param corrupt_func_X Optional function applied to the training portion
#'   of each fold before the pipeline and model fitting; for noise-robustness
#'   experiments. Signature \code{function(X_train, fold_i)}: \code{fold_i}
#'   is the fold number (1-based; \code{0} during refit). Must return a
#'   matrix with the same dimensions as the input. Test folds are never
#'   contaminated. Default is \code{NULL}.
#' @param corrupt_func_y Optional function applied to the training response
#'   of each fold before the pipeline and model fitting; same contract as
#'   \code{corrupt_func_X}. Default is \code{NULL}.
#'
#' @return A \code{matrix} of size \code{length(metrics) x length(folds)}.
#'   Row names correspond to metric names; column \code{i} contains the
#'   metric values computed on fold \code{i}.
#'
#' @details
#' Preprocessing pipelines (\code{pipeline_X} and \code{pipeline_y}) are fit
#' on the training portion of each fold and then applied to both training and
#' test portions. This prevents data leakage from the test set into the
#' training pipeline. Contamination functions (\code{corrupt_func_X},
#' \code{corrupt_func_y}) run after the split and before the pipelines, so
#' noise levels defined from training-fold statistics use raw (unscaled)
#' values.
#'
#' @examples
#' \dontrun{
#' # Example: linear regression with RMSE
#' rmse <- function(y, y_hat) sqrt(mean((y - y_hat)^2))
#'
#' my_lm <- function(X, y) {
#'   lm.fit(cbind(1, X), y)
#' }
#'
#' my_predict <- function(fit, X) {
#'   cbind(1, X) %*% fit$coefficients
#' }
#'
#' cv_res <- K_fold_cross_validation(
#'   model = my_lm,
#'   X = mtcars[, c("wt", "disp")],
#'   y = mtcars$mpg,
#'   folds = make_k_folds(nrow(mtcars), 5),
#'   metrics = list(rmse = rmse),
#'   predict_func = my_predict,
#'   model_seed = 42
#' )
#'
#' print(cv_res)
#' rowMeans(cv_res)  # average RMSE across folds
#' }
#'
#' @importFrom stats predict
#' @importFrom stats sd
#' @seealso \code{\link{make_k_folds}}, \code{\link{grid_search_cv}}
#' @export
K_fold_cross_validation <- function(
  model,
  X,
  y,
  folds,
  metrics,
  predict_func = predict,
  pipeline_X = NULL,
  pipeline_y = NULL,
  metrics_params = NULL,
  predict_params = NULL,
  model_settings = NULL,
  model_seed = NULL,
  corrupt_func_X = NULL,
  corrupt_func_X_seeds = NULL,
  corrupt_func_y = NULL,
  corrupt_func_y_seeds = NULL
) {

  args <- list(
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
    model_settings = model_settings,
    corrupt_func_X = corrupt_func_X,
    corrupt_func_X_seeds = corrupt_func_X_seeds,
    corrupt_func_y = corrupt_func_y,
    corrupt_func_y_seeds = corrupt_func_y_seeds
  )

  if (!is.null(model_seed)) {
    withr::with_seed(model_seed, do.call(.K_fold_cross_validation_impl, args))
  } else {
    do.call(.K_fold_cross_validation_impl, args)
  }
}

.K_fold_cross_validation_impl <- function(
  model,
  X,
  y,
  folds,
  metrics,
  predict_func = predict,
  pipeline_X = NULL,
  pipeline_y = NULL,
  metrics_params = NULL,
  predict_params = NULL,
  model_settings = NULL,
  corrupt_func_X = NULL,
  corrupt_func_X_seeds = NULL,
  corrupt_func_y = NULL,
  corrupt_func_y_seeds = NULL
) {
  X <- as.matrix(X)
  y <- as.matrix(y)

  num_folds      <- length(folds)
  metrics        <- metrics_check_cv(metrics)
  num_metric     <- length(metrics)
  metrics_params <- metrics_params_check_cv(num_metric, metrics_params)
  metric_mat     <- matrix(NA_real_, nrow = num_metric, ncol = num_folds)
  fold_errors    <- vector("list", num_folds)

  for (i in seq_len(num_folds)) {
    fold_errors[[i]] <- tryCatch({
      idx     <- folds[[i]]

      X_test  <- X[idx, , drop = FALSE]
      y_test  <- y[idx, , drop = FALSE]

      X_train <- X[-idx, , drop = FALSE]
      y_train <- y[-idx, , drop = FALSE]

      if (!is.null(corrupt_func_X)) {
        X_train <- corrupt_func_X(X_train, corrupt_func_X_seeds[i])
      }
      if (!is.null(corrupt_func_y)) {
        y_train <- corrupt_func_y(y_train, corrupt_func_y_seeds[i])
      }

      if (!is.null(pipeline_X)) {
        for (pipi in seq_along(pipeline_X)) {
          pip_temp <- pipeline_X[[pipi]](X_train)
          X_train  <- trans(pip_temp, X_train)
          X_test   <- trans(pip_temp, X_test)
        }
      }

      if (!is.null(pipeline_y)) {
        for (pipi in seq_along(pipeline_y)) {
          pip_temp <- pipeline_y[[pipi]](y_train)
          y_train  <- trans(pip_temp, y_train)
          y_test   <- trans(pip_temp, y_test)
        }
      }

      model_res <- do.call(
        model,
        append(list(X = X_train, y = y_train), model_settings)
      )

      predict_res <- predict_model(
        model_res,
        X_test,
        y_test,
        predict_params,
        predict_func
      )

      for (j in seq_len(num_metric)) {
        metric_mat[j, i] <- metric_evaluate(
          metrics[[j]],
          y_test,
          predict_res$y_test_hat,
          metrics_params[[j]]
        )
      }
    }, error = function(e) {
      warning(sprintf("Fold %d failed: %s", i, conditionMessage(e)))
      list(fold = i, message = conditionMessage(e))}
    )
  }

  rownames(metric_mat) <- names(metrics)
  metric_mat <- structure(
    metric_mat,
    class        = c("crossval", "matrix"),
    num_folds    = num_folds,
    metric_names = names(metrics)
  )
  metric_mat
}

#' Print Cross-Validation Results
#'
#' Prints a header with run information and a summary matrix with the
#' mean and sd of each metric across folds.
#'
#' @param x A \code{crossval} object.
#' @param digits Number of digits to display. Defaults to 4.
#' @param ... Unused.
#' @export
print.crossval <- function(x, digits = 4, ...) {
  metric_names <- rownames(x)

  cli::cli_h1("K-Fold Cross Validation")
  info_names <- c("Number of folds", "Number of metrics")
  info_vals  <- c(as.character(attr(x, "num_folds")),
                  as.character(length(metric_names)))
  pad <- max(nchar(info_names))
  cli::cli_verbatim(sprintf(paste0("%-", pad, "s: %s"), info_names, info_vals))

  cli::cli_h2("Summary of Metrics")
  score_mat <- cbind(mean = rowMeans(x, na.rm = TRUE),
                     sd = apply(x, 1, sd, na.rm = TRUE))
  rownames(score_mat) <- metric_names
  mat_lines <- capture.output(
    print(format(round(score_mat, digits), nsmall = digits), quote = FALSE, right = TRUE)
  )
  cli::cli_verbatim(mat_lines)
  cat("\n")
  invisible(x)
}
