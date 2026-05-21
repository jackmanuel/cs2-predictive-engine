# Model Evaluation

This project uses walk-forward validation to evaluate feature changes before
promoting them into the production training pipeline. The goal is to compare
feature sets on the same future match windows, rather than judging each training
run by its own moving holdout split.

Generated CSVs and HTML/Markdown reports are local artefacts under `reports/`
and are not committed to the repository. The tracked source of truth is the
evaluation process itself.

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

## Post-Hoc Calibration

Each fold also runs a simple post-hoc shrinkage calibration diagnostic. After
the neural net is trained, the runner saves predictions on the inner validation
split, chooses the shrink factor that minimizes inner validation log loss, and
then applies that factor to the outer evaluation predictions:

```text
p_calibrated = 0.5 + shrink * (p_raw - 0.5)
```

The default candidates are:

```text
0.70 0.80 0.90 1.00
```

Override them with:

```bash
python -m evaluation.feature_experiments --calibration-shrink-values 0.70 0.80 0.90 1.00 1.10
```

This is deliberately fitted only on inner validation data. The outer
walk-forward fold is still untouched until final evaluation, so calibrated
Brier/log loss/ECE are comparable without future leakage. ROC AUC is reported
separately because calibration should not usually improve ranking quality.

## Metrics

- **Brier score:** mean squared error of predicted probabilities. Lower is
  better. This is especially useful for calibration.
- **Log loss:** probability error that heavily punishes confident wrong
  predictions. Lower is better, and this is important for betting-style models.
- **Accuracy:** share of maps where the `>= 0.5` side wins. Higher is better,
  but this ignores confidence.
- **ROC AUC:** ranking quality. Higher is better. A model can have good AUC but
  poor probability calibration.
- **ECE:** expected calibration error across probability bins. Lower is better;
  it is a compact view of average calibration gap.

The summary CSV also includes deltas versus `baseline_all`. Negative Brier/log
loss deltas mean a variant improved on the current feature set.

## Production Training Diagnostics

Production retrains still use a temporal 70/15/15 train/validation/test split
inside `model.train`. The version row keeps the historical five-seed average
test Brier and log loss for continuity, while the selected saved checkpoint is
now evaluated separately in `model_version_evaluations`.

Each registered model version can have named evaluation rows:

- `temporal_test`: the selected archived checkpoint on that version's temporal
  held-out test slice.
- `mirrored_temporal_test`: the same test rows with Team A/B swapped and labels
  flipped, used to detect ordering-sensitive behaviour.
- `order_symmetry`: the mean and 95th-percentile value of
  `abs(P(original Team A wins) + P(mirrored Team B wins) - 1)`.

These rows are stored in the shadow-ledger SQLite database and shown in the
dashboard's Model tab. The dashboard focuses on selected-checkpoint metrics:
Brier score, log loss, mirrored log loss, accuracy, ROC AUC, label mean,
prediction mean, symmetry error, training maps, and the reconstructed test
slice. Rank-ordering diagnostics are retained in the backend for investigation
but are not surfaced as headline dashboard columns.

## Current Feature Hypotheses

These notes are working hypotheses, not permanent conclusions.

- `rank_diff` is the core team-strength proxy and should be treated as a
  baseline feature. A ranking-source experiment compared the production
  `world_rank` with VRS fallback against pure world rank, pure VRS rank,
  averaged world/VRS ranks, weighted blends, and different unknown-rank
  defaults. The confirmation run found that averaging
  available world/VRS ranks with an unknown default of 750 was the strongest
  experimental variant, but the simpler production-compatible choice of keeping
  world rank with VRS fallback and changing the default from 500 to 750 also
  improved Brier score and log loss versus `baseline_all`.
- Unknown or unranked teams now default to rank 750.
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

## Related Reports

The project currently has two standalone report artefacts:

- `model.automate_predictions` writes the upcoming-match prediction report as a
  standalone HTML file.
- `evaluation.shadow_ledger report` writes the model performance and calibration
  report as a standalone HTML file.

`dashboard_server.py` provides a local dashboard around those artefacts. It
serves the latest reports, exposes a manual prediction playground, can run the
`update.py` scraper workflow once or in its jittered loop, can launch the full
`pipeline.py` retrain job, and shows model-version training diagnostics in the
Model tab. The scraper dashboard tracks separate completion times for
recent-results scraping, upcoming-match/odds scraping, and shadow-ledger
refreshes because those stages can finish at meaningfully different times.

## Promotion Rule

Do not promote a feature change based on one run or one fold. A candidate should
beat `baseline_all` across multiple folds and seeds, with emphasis on Brier
score and log loss. AUC and accuracy are useful secondary checks, but calibrated
probabilities are the main target.

Generated experiment CSVs are written under `reports/` and are intentionally not
required as source files. The repo should preserve the evaluation process; each
clone can re-run the experiments against its own current corpus.
