# EazyCV

**Flexible K-fold cross-validation for arbitrary models, in plain R.**

EazyCV is a lightweight framework that turns *any* model-fitting function,
prediction routine, and evaluation metric into a reproducible cross-validation
experiment. It ships with preprocessing pipelines, per-metric hyperparameter
search, nested cross-validation, and noise-robustness (contamination) studies —
without asking you to adopt a modeling API.

- **Model agnostic** — pass a closure, not a class.
- **Metric agnostic** — pass `function(y, y_hat)`; results are kept per metric.
- **Reproducible** — every random stream is seeded, and the global RNG is left
  untouched.
- **Leak-free** — pipelines are fitted on the training portion of each fold only.

---

## Installation

```r
pak::pak("define957/EazyCV")
```

## Quick start

```r
library(EazyCV)

rmse <- function(y, y_hat) sqrt(mean((y - y_hat)^2))

my_lm    <- function(X, y, shift = 0) lm.fit(cbind(1, X), y + shift)
my_predict <- function(fit, X) cbind(1, X) %*% fit$coefficients

X     <- mtcars[, c("wt", "disp")]
y     <- mtcars$mpg
folds <- make_k_folds(nrow(mtcars), K = 5, shuffle = TRUE, seed = 1)

# 1. Cross-validation
cv <- K_fold_cross_validation(
  model = my_lm, X = X, y = y, folds = folds,
  metrics = list(rmse = rmse), predict_func = my_predict,
  model_seed = 42
)

# 2. Grid search over hyperparameters
gs <- grid_search_cv(
  model = my_lm, X = X, y = y, folds = folds,
  metrics = list(rmse = rmse), param_list = list(shift = c(0, 0.5, 1)),
  predict_func = my_predict, select_direction = "min",
  model_seed = 42, threads_num = 2
)

# 3. Refit on the full data and evaluate
refit <- refit_model(gs, refit_seed = 7)
evaluate_refit(gs, X, y, refit = refit)
```

## Table of contents

- [Core workflow](#core-workflow)
- [Metrics](#metrics)
- [Nested cross-validation](#nested-cross-validation)
- [Preprocessing pipelines](#preprocessing-pipelines)
- [Contamination experiments](#contamination-experiments)
- [Random seeds](#random-seeds)
- [Notes](#notes)
- [License](#license)

## Core workflow

| Function | Purpose |
| --- | --- |
| `make_k_folds()` | Build a list of held-out row indices, one per fold. |
| `K_fold_cross_validation()` | Fit the model on each training portion, predict the held-out fold, and return a metrics-by-folds matrix. |
| `grid_search_cv()` | Expand `param_list` with `expand.grid()`, run CV for every candidate in parallel, and select the best candidate per metric. |
| `refit_model()` | Fit one final model per metric on the full data using that metric's best parameters. |
| `evaluate_refit()` | Apply the stored transformers to new data, predict, and compute the metrics. |
| `nested_cv()` | Nest a whole grid search inside an outer fold loop, so hyperparameters are selected without ever touching the fold used for scoring. |

## Metrics

A metric is any function of the form `function(y, y_hat)`. Pass a named list of
them through `metrics`; every metric gets its own column in the results, and
`grid_search_cv()` selects the best candidate **per metric**, so different
columns may come from different configurations.

### Built-in metrics

Built-in metrics cover both classification and regression. Classification
metrics take the predicted class labels and the true class labels, and they
expect a factor or character vector of labels; regression metrics take numeric
observed and predicted values.

#### Classification

| Function | Description |
| --- | --- |
| `accuracy()` | Fraction of predictions that exactly match the ground truth. |
| `balanced_accuracy()` | Unweighted mean of the per-class recall. With `adjusted = TRUE`, rescaled so that random performance scores 0 and perfect performance scores 1. |
| `geometric_mean_of_recall()` | Geometric mean of the per-class recall (G-mean). Multiplies the class-wise sensitivities, so a single failing class pulls the whole score down. Supports `sample_weight` and a `correction` value for classes with no true samples. |

```r
cv <- K_fold_cross_validation(
  model = my_classifier, X = X, y = y, folds = folds,
  metrics = list(
    acc   = accuracy,
    bacc  = balanced_accuracy,
    gmean = geometric_mean_of_recall
  ),
  predict_func = my_predict
)
```

#### Regression

| Function | Description |
| --- | --- |
| `mean_absolute_error()` | Mean of the absolute residuals, `mean(abs(y - y_hat))`. Expressed in the original units of `y`; lower is better. |
| `mean_squared_error()` | Mean of the squared residuals, `mean((y - y_hat)^2)`. Expressed in the squared units of `y` and sensitive to outliers; lower is better. |
| `root_mean_squared_error()` | Square root of MSE, `sqrt(mean_squared_error(y, y_hat))`. Expressed in the original units of `y`; lower is better. |
| `symmetric_mean_absolute_percentage_error()` | Symmetric mean absolute percentage error (SMAPE), the mean of `2 * abs(y - y_hat) / (abs(y) + abs(y_hat))` in `[0, 2]`. Lower is better; multiply by 100 for a percentage. |

```r
cv <- K_fold_cross_validation(
  model = my_lm, X = X, y = y, folds = folds,
  metrics = list(
    mae   = mean_absolute_error,
    mse   = mean_squared_error,
    rmse  = root_mean_squared_error,
    smape = symmetric_mean_absolute_percentage_error
  ),
  predict_func = my_predict
)
```

### Custom metrics

Any user-defined function works:

```r
r2 <- function(y, y_hat) {
  1 - sum((y - y_hat)^2) / sum((y - mean(y))^2)
}

metrics = list(r2 = r2)
```

Extra arguments can be supplied per metric through `metrics_params`.

## Nested cross-validation

`grid_search_cv()` reports the CV score of the best candidate, which is
optimistically biased: the same folds both selected and scored it.
`nested_cv()` adds an outer loop. Hyperparameters are selected on the outer
training set only, the chosen model is refitted there, and it is scored once
on the untouched outer test fold. The result estimates the whole tuning
procedure, not one pre-selected configuration.

```r
outer <- make_k_folds(nrow(mtcars), K = 3, shuffle = TRUE, seed = 1)

scores <- nested_cv(
  model = my_lm, X = X, y = y, outer_folds = outer,
  metrics = list(rmse = rmse), param_list = list(shift = c(0, 0.5, 1)),
  predict_func = my_predict, select_direction = "min",
  inner_folds_num   = 3,
  inner_folds_seeds = c(11, 12, 13),
  model_seeds       = c(101, 102, 103),
  refit_seeds       = c(201, 202, 203),
  threads_num       = 2
)

scores            # one row per outer fold, one column per metric
colMeans(scores)  # nested-CV estimate of the tuning procedure
```

- Each outer fold runs its own grid search, so the cost is
  `outer folds x grid size x inner folds` model fits, plus one refit per
  outer fold and metric.
- The seed arguments are vectors with one element per outer fold. The
  contamination seeds are a list whose element `i` holds `inner_folds_num`
  seeds. Supplying all of them makes the run reproducible and leaves the
  global RNG untouched.
- Selection is per metric, so different columns of the result may come from
  different configurations and different fitted models.
- `outer_folds` must partition the data; generate it with `make_k_folds()`
  (`stratified = TRUE` for classification).

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
| `model_seeds` (`nested_cv()`) | one model seed per outer fold; inner fold `k` is seeded with `model_seeds[i] + k` |
| `inner_folds_seeds`, `refit_seeds` (`nested_cv()`) | inner split seed and refit seed, one per outer fold |
| `corrupt_func_X_seeds_list`, `corrupt_func_y_seeds_list` (`nested_cv()`) | one vector of `inner_folds_num` contamination seeds per outer fold |

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
- See `?K_fold_cross_validation`, `?grid_search_cv`, `?nested_cv`,
  `?refit_model`, `?evaluate_refit`, `?accuracy`, `?balanced_accuracy`,
  `?geometric_mean_of_recall`, `?mean_absolute_error`,
  `?mean_squared_error`, `?root_mean_squared_error`,
  `?symmetric_mean_absolute_percentage_error` for details.

## License

See `LICENSE`.
