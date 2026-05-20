# R Robust Regression Evolution

This example evolves R programs for a synthetic robust linear-regression benchmark. The evaluator now computes all metrics in Python on deterministic train/test splits across multiple seeds, so candidates are scored on held-out generalization and coefficient recovery rather than self-reported in-sample metrics.

## Files

- `initial_program.r`: starting R implementation using ordinary least squares with intercept
- `evaluator.py`: Python evaluator that generates synthetic datasets, runs the R program, and computes benchmark metrics
- `config.yaml`: evolution config for the robust-regression benchmark
- `requirements.txt`: Python dependencies for the evaluator

## Candidate Contract

The evolved R program should expose this interface:

```r
fit_model <- function(X_train, y_train) {
  ...
  list(coefficients = as.numeric(coefficients))
}

predict_model <- function(model, X_test) {
  ...
  as.numeric(predictions)
}

main <- function(X_train, y_train, X_test) {
  model <- fit_model(X_train, y_train)
  predictions <- predict_model(model, X_test)
  list(
    coefficients = as.numeric(model$coefficients),
    predictions = as.numeric(predictions)
  )
}
```

The evaluator ignores self-reported `mse`, `mae`, `r_squared`, `outlier_robustness`, or similar fields.

## Benchmark Shape

- Train/test splits are explicit.
- Stage 1 uses `STAGE1_SEEDS = (0, 1)`.
- Full evaluation uses `FULL_EVAL_SEEDS = (0, 1, 2, 3, 4)`.
- Selection is driven by `combined_score = score`.
- Runtime is recorded diagnostically but is not part of the objective.

The four task IDs used by MT-STS are:

- `rr_outliers10_100x3`
- `rr_outliers20_100x3`
- `rr_leverage10_100x3`
- `rr_hard_120x8`

## Prerequisites

Install R and `jsonlite`:

```r
install.packages(c("jsonlite"))
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the standalone evaluator on the baseline program:

```bash
python examples/r_robust_regression/evaluator.py \
  examples/r_robust_regression/initial_program.r
```

Run OpenEvolve on the full shared robust-regression family:

```bash
cd examples/r_robust_regression
python ../../openevolve-run.py initial_program.r evaluator.py --config config.yaml --iterations 100
```

Run one specific task variant:

```bash
R_ROBUST_TASK_ID=rr_outliers10_100x3 \
python examples/r_robust_regression/evaluator.py \
  examples/r_robust_regression/initial_program.r
```

For the full multi-task shared-archive-then-specialize workflow, see
[multi_task_shared_then_adapt/docs/README.md](/home/alperen/llm-opt/openevolve/multi_task_shared_then_adapt/docs/README.md).
