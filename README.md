# CS2 Predictive Engine

A high-performance Python framework for Counter-Strike 2 match prediction. This engine combines **Monte Carlo Veto Simulations** with a **PyTorch Neural Network** to calculate precise series win probabilities.

---

## Key Features

- **Native HLTV Ingestion:** High-fidelity scraping of round histories, player metrics, and map vetoes using Selenium.
- **Monte Carlo Veto Simulator:** Simulates thousands of veto paths to model map pool variance and "permaban" bluffing.
- **Neural Map Predictor:** A PyTorch-based binary classifier trained on temporal features (rolling win rates, streaks, H2H).
- **Automated Match Dashboard:** Scrapes upcoming HLTV matches and generates premium HTML reports with integrated betting odds and map visuals.
- **Zero Future Leakage:** Temporal feature engineering ensures models are only trained on data available *at the time of the match*.

---

## Architecture

1.  **Ingestion Layer:** `hltv_client.py` handles complex interactions with HLTV, bypassing protections to build a canonical match database.
2.  **Simulation Layer:** `veto_sim.py` models team banning/picking behavior using historical bias and Laplace smoothing.
3.  **Model Layer:** `net.py` (Architecture) + `predict.py` (Inference logic) use mirrored data samples to eliminate positional bias.
4.  **Reporting Layer:** `automate_predictions.py` orchestrates the end-to-end flow from live scraping to HTML dashboard generation.

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
Run the full pipeline to ingest, clean, engineer features, and train the model:
```bash
python pipeline.py
```

### 2. Live Match Automation (The Dashboard)
Scrape upcoming matches and generate a premium HTML dashboard with model vs. market probability comparisons:
```bash
# Predict matches for a specific HLTV Event
python model/automate_predictions.py --event-id 8242 --output rio_results.html

# Predict all upcoming matches
python model/automate_predictions.py
```

### 3. Manual Series Prediction
Predict a specific hypothetical or scheduled matchup:
```bash
python model/predict_series.py "Vitality" "G2" --format bo3
```

---

## Technical Components

| Component | Description |
| :--- | :--- |
| **Ingestion** | Selenium + BeautifulSoup4 |
| **Model** | PyTorch Neural Network (Binary Classifier) |
| **Veto** | Monte Carlo Simulation (10,000+ iterations) |
| **Features** | Temporal rolling windows, Mirroring, Rank Differentials |
| **Backend** | Parquet data storage for high-speed I/O |
