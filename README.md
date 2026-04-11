# CS2 Predictive Engine

A modular Python framework for ingesting Counter-Strike 2 match data, engineering temporal features, and training a predictive model for match outcomes to find Expected Value (EV) opportunities. 

Currently implemented as a Minimum Viable Product (MVP) using a PyTorch binary classifier on tabular features.

## Architecture

1. **Ingestion Layer:** Scraping HLTV natively to serve as the highly-detailed "Canonical Match Database" (with round histories, player metrics, map vetoes, and match summaries).
2. **Processing Layer:** Cleaners and Transformers operate on the HLTV dataset. Engineers temporal features (rolling win rates, current streaks, Head-to-Head records). Operates chronologically to absolutely guarantee no future data leakage.
3. **Model Layer:** A PyTorch neural network `MatchPredictor` with a custom `Dataset`. Designed to be easily swappable with more complex architectures (like PyTorch entity embeddings) in future scope.
4. **Evaluation Layer:** Walk-forward backtesting system evaluating the model's accuracy, log-loss, and Brier score (crucial for probability calibration).

## Setup & Installation

1. Create a virtual environment and install dependencies:
   ```bash
   # Linux/macOS
   python3 -m venv venv
   source venv/bin/activate

   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env`.
   ```bash
   cp .env.example .env
   ```

## Usage

Run the modules sequentially:

1. **Scrape Canonical Match Data (HLTV):**
   *(Scrapes detailed round, player, ranking, and map info natively. Use `--matches` and `--pages` to batch safely.)*
   ```bash
   python -m ingestion.fetch_hltv_matches --pages 5 --matches 50
   ```

2. **[DEPRECATED] Fetch Contextual Match Data (PandaScore):**
   *PandaScore integration is deprecated and no longer used in the main pipeline.*

3. **Run Unified Pipeline (Clean, Feature, Train):**
   Instead of running steps manually, you can run the whole pipeline after you've ingested new data:
   ```bash
   python pipeline.py
   ```

   This sequentially runs the cleaning, feature engineering, and model training modules.

4. **Individual Steps (Optional):**
   If you need to run specific parts of the pipeline:
   - **Clean & Parse Data:** `python -m processing.clean`
   - **Engineer Temporal Features:** `python -m processing.features`
   - **Train the MVP Model:** `python -m model.train`

6. **Evaluate (Backtest) on Held-Out Data:**
   ```bash
   python -m evaluation.backtest
   ```

## Future Scope
- Sequential map veto probability trees.
- LLM agent swarm for NLP sentiment analysis (parsing Reddit/Twitter for roster rumours).
- Advanced PyTorch entity embeddings for individual players.
- Integration with external odds APIs (e.g., The Odds API) for real EV calculation.
