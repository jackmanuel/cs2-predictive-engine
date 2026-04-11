"""
Central configuration for the CS2 Predictive Engine.
Loads environment variables and defines paths, API settings, and hyperparameters.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"

# Create directories on import
for d in [RAW_DIR, PROCESSED_DIR, CHECKPOINT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- PandaScore API (DEPRECATED: Use HLTV only) ---
# PANDASCORE_API_KEY = os.getenv("PANDASCORE_API_KEY")
# PANDASCORE_BASE_URL = "https://api.pandascore.co"
# PANDASCORE_CSGO_PREFIX = "/csgo"
# RATE_LIMIT_PER_HOUR = 1000
# REQUEST_DELAY_S = 3.6
# MAX_RETRIES = 3
# DEFAULT_PAGE_SIZE = 100

# --- Feature Engineering ---
MIN_MATCHES_THRESHOLD = 1   # Only include teams with at least this many matches
ROLLING_WINDOW_DAYS = 30    # Window for recent-form features
DEFAULT_TEAM_RANK = 500     # Fallback for unranked or new teams

# --- Model Hyperparameters ---
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
DROPOUT_RATE = 0.3

# Train/Val/Test split ratios (temporal, not random)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
