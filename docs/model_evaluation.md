# Model Evaluation

This project uses walk-forward validation to evaluate feature changes before
promoting them into the production training pipeline. The goal is to compare
feature sets on the same future match windows, rather than judging each training
run by its own moving holdout split.

## Experiment Runner

Run the feature experiment harness with:

```bash
python -m evaluation.feature_experiments
```

The default run uses:

- `--preset promising`
- `--folds 4`
- `--fold-days 7`
- `--seeds 1 2 3 4 5`
- `--epochs 100`
- `--patience 10`
- `--inner-val-ratio 0.15`

That produces:

```text
8 feature variants x 4 folds x 5 seeds = 160 model trainings
```

For a more stable evaluation window, prefer 14-day folds:

```bash
python -m evaluation.feature_experiments --fold-days 14 --folds 4
```

For a broader feature-ablation suite:

```bash
python -m evaluation.feature_experiments --preset full --fold-days 14 --folds 4 --seeds 1 2 3 4 5
```

Preview any run without training:

```bash
python -m evaluation.feature_experiments --dry-run
```

By default, each completed run writes timestamped CSVs under `reports/`, for
example:

```text
reports/feature_experiment_promising_14d_4folds_5seeds_20260505_193147_results.csv
reports/feature_experiment_promising_14d_4folds_5seeds_20260505_193147_summary.csv
```

Use `--run-name` to choose the report stem while still getting separate results
and summary files:

```bash
python -m evaluation.feature_experiments --fold-days 14 --run-name h2h_window_check
```

Use `--output` and `--summary-output` only when you intentionally want exact
paths.

## How Folds Work

A fold is a fixed temporal train/evaluate split. For example:

```text
train: everything before 2026-04-24
eval:  2026-04-24 through 2026-04-30
```

Each feature variant is trained and evaluated on the same folds, so the results
are comparable. The outer evaluation window is never used for early stopping.
Within each fold's training period, the runner reserves the most recent
`--inner-val-ratio` portion as an inner validation set for early stopping.

## Metrics

- **Brier score:** mean squared error of predicted probabilities. Lower is
  better. This is especially useful for calibration.
- **Log loss:** probability error that heavily punishes confident wrong
  predictions. Lower is better, and this is important for betting-style models.
- **Accuracy:** share of maps where the `>= 0.5` side wins. Higher is better,
  but this ignores confidence.
- **ROC AUC:** ranking quality. Higher is better. A model can have good AUC but
  poor probability calibration.

The summary CSV also includes deltas versus `baseline_all`. Negative Brier/log
loss deltas mean a variant improved on the current feature set.

## Current Feature Hypotheses

These notes are working hypotheses, not permanent conclusions.

- `rank_diff` is the core team-strength proxy and should be treated as a
  baseline feature.
- `win_rate_90d_diff`, `win_rate_30d_diff`, and `win_rate_7d_diff` measure form
  at different horizons. The 7-day feature may be noisy, but it uses smoothed
  win rates rather than a zero baseline.
- `h2h_a_wins_30d` and `h2h_b_wins_30d` represent prior head-to-head map counts
  within the last 30 days. This replaced all-time H2H counts after walk-forward
  experiments showed the 30-day variant was more promising.
- `sos_90d_diff` and `sos_diff` are theoretically valuable because raw win rates
  do not account for opponent strength. The current implementation is an average
  opponent-rank proxy, so it may be improved later with performance above
  expectation.
- `lan_rate_diff` may act as a team-tier proxy rather than a true match-context
  feature because the model does not currently receive `is_lan` as an input.
- `dominance_diff` and `resilience_diff` may be noisy because they are raw
  round-margin summaries and are not opponent-adjusted.

## Promotion Rule

Do not promote a feature change based on one run or one fold. A candidate should
beat `baseline_all` across multiple folds and seeds, with emphasis on Brier
score and log loss. AUC and accuracy are useful secondary checks, but calibrated
probabilities are the main target.

Generated experiment CSVs are written under `reports/` and are intentionally not
required as source files. The repo should preserve the evaluation process; each
clone can re-run the experiments against its own current corpus.
