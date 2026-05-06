# CS2 Predictive Engine

A high-performance Python framework for Counter-Strike 2 match prediction. This engine combines **Monte Carlo Veto Simulations** with a **PyTorch Neural Network** to calculate precise series win probabilities.

---

## Key Features

- **Native HLTV Ingestion:** High-fidelity scraping of round histories, player metrics, and map vetoes using Selenium.
- **Monte Carlo Veto Simulator:** Simulates thousands of veto paths to model map pool variance and "permaban" bluffing.
- **Neural Map Predictor:** A PyTorch-based binary classifier trained on temporal features (rolling win rates, streaks, H2H).
- **Automated Match Dashboard:** Scrapes upcoming HLTV matches and generates premium HTML reports with integrated betting odds and map visuals.
- **Shadow Ledger Performance Report:** SQLite-backed paper-trading system with an interactive HTML report for calibration bins, match browsing, model version comparison, odds, and paper-trading returns.
- **Model Version Registry:** Every training run is archived with its weights, features, hyperparameters, and architecture hash for full reproducibility.
- **Zero Future Leakage:** Temporal feature engineering ensures models are only trained on data available *at the time of the match*.

---

## Architecture

1.  **Ingestion Layer:** `hltv_client.py` handles complex interactions with HLTV, bypassing protections to build a canonical match database.
2.  **Processing Layer:** `clean.py` normalizes raw data; `features.py` computes temporal differentials with configurable rolling windows.
3.  **Simulation Layer:** `veto_sim.py` models team banning/picking behaviour using historical bias and Laplace smoothing.
4.  **Model Layer:** `net.py` (Architecture) + `predict.py` (Inference) use mirrored data samples to eliminate positional bias.
5.  **Reporting Layer:** `automate_predictions.py` orchestrates end-to-end flow from live scraping to HTML dashboard generation.
6.  **Evaluation Layer:** `shadow_ledger.py` (SQLite calibration tracker and HTML performance report) + `backtest.py` (held-out test evaluation).

---

## Setup & Installation

1. **Environment Setup:**
   ```bash
   # Create and activate venv
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   source venv/bin/activate # Linux/macOS

   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Asset Setup:**
   To enable map visuals in reports, place map PNGs in `static/maps/` (e.g., `de_dust2.png`). These are ignored by git for copyright safety.

---

## Usage

### 1. Unified Pipeline (Training)
Run the full pipeline to ingest, clean, engineer features, and train the model. Each run registers a new model version and archives the weights:
```bash
python pipeline.py
```

### 2. Daily Update
Run the lightweight update workflow without retraining. By default, this runs in a loop about every two hours, with random jitter added between passes. Each pass scrapes the first page of recent HLTV results, checks whether the current model is at least 500 cleaned maps behind the canonical scrape, generates the prediction report, and refreshes the shadow ledger:
```bash
python update.py --no-open
```

The freshness check reuses the same cleaning exclusions as training, so forfeits/defaults and other excluded map rows are not counted as model lag. If the warning triggers, it only prints to the console; `update.py` never retrains the model.

Useful options:
```bash
# Run a single update pass and exit
python update.py --run-once

# Scrape more recent-results pages before predicting
python update.py --pages 3

# Limit scraping to a fixed number of new matches
python update.py --matches 20

# Generate predictions for a specific HLTV event
python update.py --event-id 8242

# Adjust the loop timing
python update.py --interval-hours 2 --jitter-minutes 45 --no-open
```

### 3. Live Match Automation (The Dashboard)
Scrape upcoming matches and generate a premium HTML dashboard. This also automatically records shadow bets for model calibration:
```bash
# Predict matches for a specific HLTV Event
python model/automate_predictions.py --event-id 8242

# Predict all upcoming matches
python model/automate_predictions.py
```

### 4. Manual Series Prediction
Predict a specific hypothetical or scheduled matchup:
```bash
python -m model.predict_series "Vitality" "G2" --format bo3
```

### 5. Shadow Ledger (Model Performance Statistics)
The shadow ledger automatically records every prediction from the dashboard. It stores the latest model probability, median market odds when available, edge calculations, model version, and eventual match result.

Generate the interactive **Model Performance Statistics** report:
```bash
# Resolve pending match results via HLTV
python -m evaluation.shadow_ledger refresh

# Generate the interactive HTML report
python -m evaluation.shadow_ledger report

# Choose a report output path
python -m evaluation.shadow_ledger report --output reports/shadow_ledger_report.html

# Print the legacy command-line summary
python -m evaluation.shadow_ledger report --text

# Show all tracked matches with latest predictions
python -m evaluation.shadow_ledger list

# Show model version history
python -m evaluation.shadow_ledger versions

# Show full odds history for a specific match
python -m evaluation.shadow_ledger odds <hltv_match_url>
```

The HTML report includes:

- **Top-level performance cards:** valid matches, settled/pending counts, model favourite accuracy, bookmaker favourite accuracy, Brier score, log loss, calibration error, and actionable edge bets.
- **Calibration bins:** compares assigned team win probabilities with actual win rates, so bins like `40-45%` show whether teams assigned that likelihood are winning at roughly that rate.
- **Edge strategy breakdowns:** flat and confidence-weighted paper-trading ROI by edge bucket, plus favourite/underdog splits and p-values.
- **Compact match browser:** searchable table of actual match results with model odds, median market odds, result, edge, book count, and snapshot count.
- **Model version history:** model versions with Brier/log loss comparison, training maps, feature counts, and matches predicted.

The report template lives at `evaluation/templates/shadow_ledger_report.html`; `shadow_ledger.py` prepares the data and injects it into the template.

### 6. Feature Experiment Runner
Run walk-forward feature ablations before promoting feature changes into the production model:
```bash
# Default promising suite: 8 variants x 4 weekly folds x 5 seeds
python -m evaluation.feature_experiments

# Inspect the planned folds and variants without training
python -m evaluation.feature_experiments --dry-run

# Fast smoke run while iterating
python -m evaluation.feature_experiments --folds 1 --seeds 1 --epochs 10 --patience 3

# Broader suite with additional exploratory ablations
python -m evaluation.feature_experiments --preset full
```

The default suite compares the current feature set against variants that remove `lan_rate_diff`, remove `dominance_diff`/`resilience_diff`, remove those features together, and test shorter/longer H2H windows around the production 30-day H2H counts. Results are written to timestamped CSVs under `reports/`; use `--run-name` or explicit output paths when you want stable names.

See `docs/model_evaluation.md` for the evaluation methodology, metric definitions, and current feature hypotheses.

### 7. Polymarket Settlement Forfeit/Default Model
Train the separate settlement-risk adjustment model. This does not retrain or replace the winner model:
```bash
python -m model.train_forfeit
```

Outputs:

- `data/checkpoints/forfeit_model.joblib` calibrated logistic regression bundle
- `data/processed/forfeit_features.parquet` leakage-safe match-level features
- `data/forfeit_training_state.json` current training-state snapshot
- `reports/forfeit_model_evaluation.md` future-held-out evaluation report
- `reports/forfeit_model_metrics.json` machine-readable metrics

Manual series predictions can print the final Polymarket fair probabilities with:
```bash
python -m model.predict_series "Vitality" "G2" --format bo3 --polymarket-adjust --event "IEM Cologne 2026" --lan
```

### 8. Betting Ledger (Real Bets)
Track real bets with model probability and edge:
```bash
# Record a bet
python betting_ledger.py add --url <hltv_url> --bookmaker "Bet365" --odds 2.10 --bet "G2" --amount 10 --model-prob 0.62

# Resolve pending results
python betting_ledger.py refresh

# View ledger
python betting_ledger.py list
```

---

## Configuration

All tuning parameters are centralised in `config.py`:

| Constant | Default | Purpose |
| :--- | :--- | :--- |
| `FORM_WINDOW_DAYS` | 30 | Rolling window for win rate, dominance, resilience, SoS |
| `FORM_WINDOW_DAYS_SHORT` | 7 | Short-term momentum window |
| `MAP_WINDOW_DAYS` | 90 | Map-specific win rate window (wider for sample size) |
| `H2H_WINDOW_DAYS` | 30 | Recent head-to-head map count window |
| `VETO_WINDOW_DAYS` | 90 | Veto simulation historical stats window |
| `MC_ITERATIONS` | 10,000 | Monte Carlo veto simulation iterations |
| `MC_THRESHOLD` | 0.90 | Cumulative probability cutoff for veto path selection |
| `WIN_STREAK_CAP` | 5 | Maximum win streak value |

---

## Technical Components

| Component | Description |
| :--- | :--- |
| **Ingestion** | Selenium + BeautifulSoup4 (undetected-chromedriver) |
| **Model** | PyTorch Neural Network (Binary Classifier, 64→32→1) |
| **Veto** | Monte Carlo Simulation (configurable iterations) |
| **Features** | Temporal rolling windows, mirroring, rank differentials, SoS |
| **Data Storage** | Parquet (training data), SQLite (shadow ledger + model registry) |
| **Versioning** | Automatic weight archival with architecture hash tracking |
