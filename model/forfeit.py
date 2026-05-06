import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHECKPOINT_DIR
from processing.forfeit_features import FORFEIT_MODEL_FEATURES, features_for_match


FORFEIT_MODEL_PATH = CHECKPOINT_DIR / "forfeit_model.joblib"


def polymarket_fair_probs(win_prob_a: float, forfeit_prob: float) -> tuple[float, float]:
    fair_a = (1 - forfeit_prob) * win_prob_a + 0.5 * forfeit_prob
    fair_b = (1 - forfeit_prob) * (1 - win_prob_a) + 0.5 * forfeit_prob
    return fair_a, fair_b


class ForfeitPredictorContext:
    """Loads the calibrated settlement-risk model and historical rate state."""

    def __init__(self, model_path: str | Path = FORFEIT_MODEL_PATH):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Forfeit model not found at {self.model_path}. Run `python -m model.train_forfeit` first."
            )

        bundle = joblib.load(self.model_path)
        self.model = bundle["model"]
        self.history_state = bundle["history_state"]
        self.metadata = bundle.get("metadata", {})

    def predict_forfeit_probability(self, match: dict[str, Any]) -> float:
        feature_df = features_for_match(match, self.history_state)
        probs = self.model.predict_proba(feature_df[FORFEIT_MODEL_FEATURES])[:, 1]
        return float(np.clip(probs[0], 0.0, 1.0))


def predict_forfeit_probability(match: dict[str, Any], ctx: ForfeitPredictorContext | None = None) -> float:
    if ctx is None:
        ctx = ForfeitPredictorContext()
    return ctx.predict_forfeit_probability(match)
