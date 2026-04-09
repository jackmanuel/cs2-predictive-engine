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

# --- PandaScore API ---
PANDASCORE_API_KEY = os.getenv("PANDASCORE_API_KEY")
PANDASCORE_BASE_URL = "https://api.pandascore.co"
# CS2 data lives under the legacy /csgo/ prefix
PANDASCORE_CSGO_PREFIX = "/csgo"

# Rate limiting (free "Fixtures" tier: ~1000 req/hr)
RATE_LIMIT_PER_HOUR = 1000
# Delay between requests in seconds (3600s / 1000 = 3.6s safe minimum)
REQUEST_DELAY_S = 3.6
MAX_RETRIES = 3

# Pagination
DEFAULT_PAGE_SIZE = 100  # PandaScore max per page

# --- Feature Engineering ---
MIN_MATCHES_THRESHOLD = 10  # Only include teams with at least this many matches
ROLLING_WINDOW_DAYS = 30    # Window for recent-form features

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
