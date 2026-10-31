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
#' @param model_seed Optional integer seed for model/pipeline randomness. If
#'   provided, the global RNG state is restored on exit and each fold
#'   \code{i} is seeded with \code{model_seed + i} before its pipeline and
#'   model fit. Contamination noise is controlled separately by
#'   \code{corrupt_func_X_seeds} / \code{corrupt_func_y_seeds}. When
#'   \code{NULL}, the global RNG is used and consumed.
#' @param corrupt_func_X Optional function applied to the training portion
#'   of each fold after the pipeline and before model fitting; for
#'   noise-robustness experiments. Signature \code{function(X_train)}. Must
#'   return a matrix with the same dimensions as the input. Test folds are
#'   never contaminated. Default is \code{NULL}.
#' @param corrupt_func_X_seeds Required when \code{corrupt_func_X} is
#'   supplied. Integer vector of length \code{length(folds)}. Fold \code{i}'s
#'   contamination runs under
#'   \code{withr::with_seed(corrupt_func_X_seeds[i], ...)}; the seed is local
#'   to that call, so the noise is independent of \code{model_seed} and does
#'   not consume the fold's model RNG stream. Use the same vector across grid
#'   candidates to compare them under identical noise.
#' @param corrupt_func_y Optional function applied to the training response
#'   of each fold after the pipeline and before model fitting; same contract
#'   as \code{corrupt_func_X}. Default is \code{NULL}.
#' @param corrupt_func_y_seeds Required when \code{corrupt_func_y} is
#'   supplied; integer vector of length \code{length(folds)}. See
#'   \code{corrupt_func_X_seeds}.
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
#' \code{corrupt_func_y}) run after the pipelines, so pipelines are fitted on
#' the clean training portion and noise is defined in the transformed units.
#' Contamination functions are always seeded explicitly: fold
#' \code{i} uses \code{corrupt_func_X_seeds[i]} and
#' \code{corrupt_func_y_seeds[i]} locally via
#' \code{withr::with_seed()}, so the noise does not depend on
#' \code{model_seed} and the fold's model stream is restored afterwards.
#' Different folds use different seeds, so fold-to-fold noise differs. Model
#' and pipeline randomness is seeded separately with \code{model_seed + i}.
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
#'
#' # contamination: flip 10% of training labels
#' # corrupt_func_y_seeds is required and must have one seed per fold; the
#' # noise is then independent of model_seed
#' cv_noisy <- K_fold_cross_validation(
#'   model = my_lm,
#'   X = mtcars[, c("wt", "disp")],
#'   y = mtcars$mpg,
#'   folds = make_k_folds(nrow(mtcars), 5),
#'   metrics = list(rmse = rmse),
#'   predict_func = my_predict,
#'   model_seed = 42,
#'   corrupt_func_y = function(y_train) {
#'     flip <- sample.int(nrow(y_train), size = floor(0.1 * nrow(y_train)))
#'     y_train[flip] <- y_train[flip] + rnorm(length(flip), sd = 5)
#'     y_train
#'   },
#'   corrupt_func_y_seeds = 1:5
#' )
#' }
#'
#' @importFrom stats predict
#' @importFrom stats sd
#' @seealso \code{\link{make_k_folds}}, \code{\link{grid_search_cv}}
#' @author Zhang Jiaqi
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

  corrupt_seeds_check_cv(corrupt_func_X_seeds, length(folds),
                         "corrupt_func_X_seeds", corrupt_func_X)
  corrupt_seeds_check_cv(corrupt_func_y_seeds, length(folds),
                         "corrupt_func_y_seeds", corrupt_func_y)

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
    model_seed = model_seed,
    corrupt_func_X = corrupt_func_X,
    corrupt_func_X_seeds = corrupt_func_X_seeds,
    corrupt_func_y = corrupt_func_y,
    corrupt_func_y_seeds = corrupt_func_y_seeds
  )

  if (!is.null(model_seed)) {
    if (!is.numeric(model_seed) || length(model_seed) != 1L || is.na(model_seed)) {
      stop("'model_seed' must be a single non-missing number.")
    }
    withr::local_preserve_seed()
  }
  do.call(.K_fold_cross_validation_impl, args)
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
  model_seed = NULL,
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
      if (!is.null(model_seed)) {
        set.seed((as.numeric(model_seed) + i) %% .Machine$integer.max)
      }
      idx     <- folds[[i]]

      X_test  <- X[idx, , drop = FALSE]
      y_test  <- y[idx, , drop = FALSE]

      X_train <- X[-idx, , drop = FALSE]
      y_train <- y[-idx, , drop = FALSE]

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

      if (!is.null(corrupt_func_X)) {
        X_train <- withr::with_seed(corrupt_func_X_seeds[i], corrupt_func_X(X_train))
      }
      if (!is.null(corrupt_func_y)) {
        y_train <- withr::with_seed(corrupt_func_y_seeds[i], corrupt_func_y(y_train))
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
    class        = c("eazycv_crossval", "matrix"),
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
#' @param x A \code{eazycv_crossval} object.
#' @param digits Number of digits to display. Defaults to 4.
#' @param ... Unused.
#' @author Zhang Jiaqi
#' @export
print.eazycv_crossval <- function(x, digits = 4, ...) {
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
