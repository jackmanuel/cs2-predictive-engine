# CS2 Predictive Engine

CS2 Predictive Engine is an experimental Counter-Strike 2 match prediction project. It combines HLTV match scraping, temporal feature engineering, a PyTorch map winner model, Monte Carlo map-veto simulation, and standalone HTML reports for upcoming-match predictions and model performance tracking.

This repository is portfolio-oriented and still evolving. The current workflow is script-driven: it generates two main standalone reports instead of running as a cohesive web application.

## Model Performance

On the current shadow-ledger evaluation set, the model favourite matched the eventual winner in 66.1% of settled evaluation-eligible matches, with a 0.2181 Brier score and 0.6272 log loss.

These results come from training on a local dataset covering roughly 7 months of historical CS2 results, approximately 6,000 matches, and 13,000+ maps. The shadow-ledger report evaluates the latest prediction snapshot per match and excludes low-sample teams, roster anomalies, forfeits, and participant changes from the headline evaluation set.

## Current State

- **Prediction report:** `model/automate_predictions.py` scrapes upcoming HLTV matches, predicts series win probabilities, estimates map-veto paths, gathers market odds when available, records shadow-ledger snapshots, and writes a standalone HTML report under `reports/`.
- **Model performance report:** `evaluation/shadow_ledger.py report` summarizes settled predictions, calibration, odds snapshots, model versions, and paper-trading style edge analysis in a standalone HTML report.
- **Training pipeline:** `pipeline.py` cleans scraped match data, rebuilds temporal features, trains the winner model, saves the scaler/checkpoint, and registers a model version.
- **Manual predictions:** `model.predict_series` predicts a matchup from inferred veto paths. `model.predict` can score a known map list from the command line when the veto or expected map pool is already known.
- **Settlement-risk adjustment:** `model.train_forfeit` trains a separate forfeit/default model used for optional Polymarket-style fair probability adjustment.

Generated data, checkpoints, reports, local map images, and environment files are intentionally ignored. A fresh clone contains the code and templates, but not the local scraped corpus or trained model artefacts.

## Legal And Data Disclaimer

This is an independent personal project and is not affiliated with, endorsed by, or sponsored by HLTV.org. The ingestion layer is built around scraping publicly available HLTV pages for personal analysis. Anyone using or extending this project is responsible for reviewing HLTV's terms, robots guidance, and any applicable laws before collecting data.

The repository does not redistribute scraped HLTV data, betting odds snapshots, generated reports, checkpoints, or copyrighted map images. For any serious or production use, prefer an official, licensed, or otherwise permitted data source.

## Features

- HLTV ingestion for match metadata, map results, veto notes, player stats, ranks, and betting analytics pages.
- Temporal feature engineering with rolling form, strength-of-schedule, head-to-head, map comfort, pick context, LAN history, and roster anomaly exclusions.
- PyTorch binary map classifier with mirrored training rows to reduce team-order bias.
- Monte Carlo veto simulator with map pool, pick/ban, permaban, and veto-starter heuristics.
- Series probability aggregation over likely veto paths.
- SQLite-backed shadow ledger for prediction snapshots, model versions, odds history, calibration, and settled-result analysis.
- Walk-forward feature experiment runner for comparing feature variants across fixed future windows.
- Separate forfeit/default settlement-risk model for optional market-specific adjustment.

## Architecture

The production winner model is a tabular neural network trained on 17 temporal map-level features:

```text
17 input features -> Linear(64) -> BatchNorm -> ReLU -> Dropout
                  -> Linear(32) -> BatchNorm -> ReLU -> Dropout
                  -> Linear(1) -> Sigmoid
```

The feature set is defined in `processing/features.py` and currently includes rank differential, 90/30/7-day form, team streaks, picker context, 30-day head-to-head counts, map win rate, map comfort, dominance/resilience, average rank tier, strength-of-schedule, and LAN-rate differential.

Series predictions are built in two stages. First, `model/veto_sim.py` simulates likely BO1/BO3/BO5 map sequences using recent team map tendencies. Then `model/predict.py` scores each map with the neural network and combines the map probabilities into a series probability.

## Repository Layout

| Path | Purpose |
| :--- | :--- |
| `ingestion/` | Selenium and BeautifulSoup HLTV scraping clients and match-history ingestion. |
| `processing/` | Raw match cleaning, exclusion rules, temporal feature engineering, and forfeit/default features. |
| `model/` | Winner model training, inference, veto simulation, report generation, and forfeit/default model training. |
| `evaluation/` | Backtests, metrics, feature experiments, forfeit analysis, and the shadow-ledger report. |
| `docs/` | Longer-form methodology notes. |
| `model/templates/` | HTML fragments for the prediction report. |
| `evaluation/templates/` | HTML template for the model performance report. |
| `static/maps/` | Optional local map images, ignored by git. |
| `data/` | Local scraped data, checkpoints, model registry, and SQLite ledger, ignored by git. |
| `reports/` | Generated HTML, CSV, JSON, and Markdown outputs, ignored by git. |

## Setup

### Prerequisites

- Python 3.10 or newer.
- Google Chrome installed locally for Selenium-based scraping.
- A fresh virtual environment is recommended.

### Installation

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

On Linux/macOS, activate with:

```bash
source venv/bin/activate
```

Optional map images can be placed in `static/maps/` using filenames such as `de_mirage.png`, `de_nuke.png`, and `de_inferno.png`. These images are ignored by git.

## Usage

### 1. Scrape Recent Results

Scrape recent finished matches into the canonical local corpus at `data/raw/hltv_matches.json`. The repository does not include match data, so this step is required before training from a fresh clone.

Model quality depends heavily on data volume. A tiny scrape is enough to test that the pipeline runs, but predictions will be poor until the corpus contains enough maps for meaningful team form, map pool, head-to-head, and veto-history features.

```bash
python -m ingestion.fetch_hltv_matches --pages 3
```

Limit the scrape to a fixed number of new matches:

```bash
python -m ingestion.fetch_hltv_matches --matches 20
```

### 2. Build Or Refresh The Training Artefacts

Run the full pipeline after local match data exists. It writes cleaned maps, engineered features, model checkpoints, a scaler, `data/training_state.json`, and a model-version registry entry.

```bash
python pipeline.py
```

### 3. Daily Update Loop

Run the lightweight update workflow without retraining. Each pass scrapes recent results, prints a model freshness warning if the current checkpoint is far behind the cleaned corpus, generates a prediction report, and refreshes the shadow ledger.

```bash
python update.py --no-open
```

Useful options:

```bash
python update.py --run-once
python update.py --pages 3
python update.py --matches 20
python update.py --event-id 8242
python update.py --interval-hours 2 --jitter-minutes 45 --no-open
```

`update.py` does not retrain the model. Retraining is still a manual `pipeline.py` step.

### 4. Prediction Report

Generate the standalone upcoming-match prediction report:

```bash
python -m model.automate_predictions --event-id 8242
```

Predict all upcoming matches from the main HLTV matches page:

```bash
python -m model.automate_predictions
```

Common options:

```bash
python -m model.automate_predictions --no-open
python -m model.automate_predictions --output reports/predictions_report.html
python -m model.automate_predictions --html-file path/to/saved_hltv_matches.html
```

### 5. Manual Series Prediction

Predict a matchup using inferred veto paths:

```bash
python -m model.predict_series "Vitality" "G2" --format bo3
```

Force the veto starter when known:

```bash
python -m model.predict_series "Vitality" "G2" --format bo3 --starts-veto "Vitality"
```

Apply the optional settlement-risk adjustment:

```bash
python -m model.predict_series "Vitality" "G2" --format bo3 --polymarket-adjust --event "IEM Cologne 2026" --lan
```

### 6. Known Map Veto Prediction

When the map veto or expected map sequence is already known, score the provided maps directly:

```bash
python -m model.predict "Vitality" "G2" --maps "Mirage,Nuke,Inferno"
```

Set the picker context for map-level probabilities:

```bash
python -m model.predict "Vitality" "G2" --maps "Mirage,Nuke,Inferno" --picker team_a
```

This path is currently command-line only and is not surfaced in the HTML prediction report.

### 7. Shadow Ledger And Model Performance Report

The shadow ledger records prediction snapshots from the prediction report, stores model and odds metadata, and resolves match results later.

```bash
python -m evaluation.shadow_ledger refresh
python -m evaluation.shadow_ledger report
```

Other commands:

```bash
python -m evaluation.shadow_ledger report --output reports/shadow_ledger_report.html
python -m evaluation.shadow_ledger report --text
python -m evaluation.shadow_ledger list
python -m evaluation.shadow_ledger versions
python -m evaluation.shadow_ledger odds <hltv_match_url>
```

The HTML report includes performance cards, calibration bins, edge strategy breakdowns, a match browser, odds history fields, and model version history.

### 8. Feature Experiments

Run walk-forward feature ablations before promoting feature changes into the production feature set:

```bash
python -m evaluation.feature_experiments
```

Useful variants:

```bash
python -m evaluation.feature_experiments --dry-run
python -m evaluation.feature_experiments --folds 1 --seeds 1 --epochs 10 --patience 3
python -m evaluation.feature_experiments --preset full --fold-days 14 --folds 4
```

See `docs/model_evaluation.md` for the evaluation methodology, metric definitions, and current feature hypotheses.

### 9. Forfeit/Default Settlement-Risk Adjustment

The winner model estimates the sporting result of a match. The optional forfeit/default adjustment uses a second machine-learning model to estimate the probability that a match has settlement-affecting default risk.

This is useful for Polymarket-style markets where some forfeits resolve both teams at 50%. When enabled, the final fair probabilities are blended toward 50/50 by the estimated forfeit/default probability.

```bash
python -m model.train_forfeit
```

## Configuration

Core tuning parameters live in `config.py`.

| Constant | Default | Purpose |
| :--- | :--- | :--- |
| `FORM_WINDOW_DAYS_LONG` | 90 | Long-term form and strength-of-schedule window. |
| `FORM_WINDOW_DAYS` | 30 | General win rate, dominance, resilience, strength-of-schedule, comfort, and LAN history window. |
| `FORM_WINDOW_DAYS_SHORT` | 7 | Short-term momentum window. |
| `MAP_WINDOW_DAYS` | 90 | Map-specific win-rate window. |
| `H2H_WINDOW_DAYS` | 30 | Recent head-to-head map count window. |
| `VETO_WINDOW_DAYS` | 90 | Veto simulation historical stats window. |
| `MC_ITERATIONS` | 10,000 | Default Monte Carlo veto simulation iterations. |
| `MC_THRESHOLD` | 0.90 | Cumulative probability cutoff for selected veto paths. |
| `TRAIN_RATIO` | 0.70 | Temporal training split ratio. |
| `VAL_RATIO` | 0.15 | Temporal validation split ratio. |

## Future Improvements

- Unify the current standalone prediction report and model performance report into a single web dashboard.
- Add a refresh control that scrapes new matches, generates predictions, updates the shadow ledger, and refreshes the report without command-line scripts.
- Add a training control that runs the cleaning, feature engineering, retraining, checkpoint registration, and evaluation pipeline from the dashboard.
- Surface known-veto prediction in the dashboard, since the model already supports command-line map-list predictions.
- Improve project cohesion so ingestion, training, prediction, and evaluation feel like one product rather than separate scripts.
- Add packaging, tests, and clearer sample-data fixtures so contributors can validate changes without a private local scrape.

## Licence

This project is licensed under the MIT License. See `LICENSE` for details.
