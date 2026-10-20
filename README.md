# EazyCV

A lightweight R package for k-fold cross-validation of arbitrary model fitting
and prediction functions, with preprocessing pipelines, per-metric
hyperparameter selection, and noise-robustness experiments.

## Installation

```r
devtools::install()
```

## Quick start

```r
library(EazyCV)

rmse <- function(y, y_hat) sqrt(mean((y - y_hat)^2))

my_lm <- function(X, y, shift = 0) lm.fit(cbind(1, X), y + shift)
my_predict <- function(fit, X) cbind(1, X) %*% fit$coefficients

X <- mtcars[, c("wt", "disp")]
y <- mtcars$mpg
folds <- make_k_folds(nrow(mtcars), K = 5, shuffle = TRUE, seed = 1)

# Cross-validation
cv <- K_fold_cross_validation(
  model = my_lm, X = X, y = y, folds = folds,
  metrics = list(rmse = rmse), predict_func = my_predict,
  model_seed = 42
)

# Grid search
gs <- grid_search_cv(
  model = my_lm, X = X, y = y, folds = folds,
  metrics = list(rmse = rmse), param_list = list(shift = c(0, 0.5, 1)),
  predict_func = my_predict, select_direction = "min",
  model_seed = 42, threads_num = 2
)

# Refit on the full data and evaluate
refit <- refit_model(gs, refit_seed = 7)
evaluate_refit(gs, X, y, refit = refit)
```

## Workflow

- `make_k_folds()` returns a list of held-out row indices, one per fold.
- `K_fold_cross_validation()` fits the model on each training portion,
  predicts the held-out fold, and returns a metrics-by-folds matrix.
- `grid_search_cv()` expands `param_list` with `expand.grid()`, runs CV for
  every candidate in parallel, and selects the best candidate per metric.
- `refit_model()` fits one final model per metric on the full data using that
  metric's best parameters.
- `evaluate_refit()` applies the stored transformers to new data, predicts,
  and computes the metrics.

## Preprocessing pipelines

`pipeline_X` and `pipeline_y` are lists of functions. Each function receives a
training matrix and returns a transformer; `trans()` applies the transformer.
Transformers are fitted on the training portion of each fold, so test data
never leaks into the fit. `refit_model()` stores the transformers fitted on
the full data so they can be reused for new observations.

```r
cv <- K_fold_cross_validation(
  ...,
  pipeline_X = list(standard_scaler)
)
```

## Contamination experiments

Contamination functions are applied to the training portion of each fold,
before the pipeline and model fit.

```r
corrupt_y <- function(y_train) {
  flip <- sample.int(nrow(y_train), size = floor(0.1 * nrow(y_train)))
  y_train[flip] <- y_train[flip] + rnorm(length(flip), sd = 5)
  y_train
}

cv_noisy <- K_fold_cross_validation(
  model = my_lm, X = X, y = y, folds = folds,
  metrics = list(rmse = rmse), predict_func = my_predict,
  model_seed = 42,
  corrupt_func_y = corrupt_y,
  corrupt_func_y_seeds = 1:5
)
```

Supplying a contamination function requires its seed vector. The vector must
have one integer seed per fold; this makes the noise at fold `i` identical
across grid-search candidates.

## Random seeds

| Argument | Controls |
| --- | --- |
| `make_k_folds(seed = )` | fold shuffle |
| `model_seed` | per-fold pipeline/model stream, seeded with `model_seed + fold index` |
| `corrupt_func_X_seeds`, `corrupt_func_y_seeds` | per-fold contamination; required with the matching function; length = number of folds |
| `refit_seed` | final refit pipeline/model stream |
| `corrupt_func_X_seed`, `corrupt_func_y_seed` | refit contamination, applied locally |

Notes:

- When `model_seed`, `refit_seed`, or `make_k_folds(seed = )` are supplied,
  the global RNG state is restored on exit. Contamination seeds are always
  applied locally and do not consume the surrounding stream.
- `grid_search_cv()` passes the same `model_seed` and contamination seed
  vectors to every candidate, so each fold position uses the same noise and
  the same model/pipeline randomness across configurations.
- Contamination seeds are applied with `withr::with_seed()`, so they do not
  consume the fold's model stream.
- Do not call `set.seed()` inside contamination functions; the package seeds
  them.
- `grid_search_cv()` workers load the installed package. Reinstall
  (`devtools::install()`) before testing changes, or workers may run an older
  version.

## Notes

- Metrics are computed on the response scale produced by `pipeline_y`.
  Without a response pipeline, that is the original scale.
- `evaluate_refit(..., refit = NULL)` refits the model on every call. Pass a
  precomputed `refit` to evaluate several datasets against the same fitted
  model, or pass `refit_seed` for reproducibility.
- See `?K_fold_cross_validation`, `?grid_search_cv`, `?refit_model`, and
  `?evaluate_refit` for details.

## License

See `LICENSE`.
