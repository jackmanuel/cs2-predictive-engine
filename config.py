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
HLTV_MATCHES_FILE = RAW_DIR / "hltv_matches.json"
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"

# Create directories on import
for d in [RAW_DIR, PROCESSED_DIR, CHECKPOINT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- Feature Engineering ---
MIN_MATCHES_THRESHOLD = 1   # Only include teams with at least this many matches
DEFAULT_TEAM_RANK = 500     # Fallback for unranked or new teams

# Rolling window periods (days) for temporal features
FORM_WINDOW_DAYS = 30       # General win rate, dominance, resilience, SoS, comfort
FORM_WINDOW_DAYS_SHORT = 7  # Short-term momentum
MAP_WINDOW_DAYS = 90        # Map-specific win rate (needs wider window for sample size)
VETO_WINDOW_DAYS = 90       # Veto sim historical stats window

WIN_STREAK_CAP = 5          # Maximum win streak value (prevents outlier streaks)
DEFAULT_SOS_RANK = 100      # Fallback opponent rank for SoS when no history exists

# --- Monte Carlo Simulation ---
MC_ITERATIONS = 10000       # Default iterations for veto simulation
MC_THRESHOLD = 0.90         # Cumulative probability cutoff for path selection

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
