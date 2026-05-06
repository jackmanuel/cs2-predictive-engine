import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHECKPOINT_DIR, DATA_DIR, PROCESSED_DIR, PROJECT_ROOT, TRAIN_RATIO, VAL_RATIO
from processing.forfeit_features import (
    CATEGORICAL_FORFEIT_FEATURES,
    FORFEIT_MODEL_FEATURES,
    FORFEIT_TARGET_COL,
    NUMERIC_FORFEIT_FEATURES,
    build_forfeit_base_rows,
    build_forfeit_feature_frame,
    build_full_history_state,
)


REPORT_PATH = PROJECT_ROOT / "reports" / "forfeit_model_evaluation.md"
METRICS_PATH = PROJECT_ROOT / "reports" / "forfeit_model_metrics.json"
FEATURES_PATH = PROCESSED_DIR / "forfeit_features.parquet"
MODEL_PATH = CHECKPOINT_DIR / "forfeit_model.joblib"
STATE_PATH = DATA_DIR / "forfeit_training_state.json"


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_pipeline(c: float) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FORFEIT_FEATURES),
            ("cat", make_one_hot_encoder(), CATEGORICAL_FORFEIT_FEATURES),
        ],
        remainder="drop",
    )
    classifier = LogisticRegression(
        C=c,
        solver="lbfgs",
        max_iter=2000,
        random_state=42,
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", classifier),
        ]
    )


def fit_prefit_calibrator(estimator: Pipeline, X_cal: pd.DataFrame, y_cal: pd.Series, method: str) -> CalibratedClassifierCV:
    try:
        from sklearn.frozen import FrozenEstimator

        calibrator = CalibratedClassifierCV(FrozenEstimator(estimator), method=method)
    except Exception:
        calibrator = CalibratedClassifierCV(estimator, method=method, cv="prefit")
    calibrator.fit(X_cal, y_cal)
    return calibrator


def split_temporally(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(["date", "match_id"]).reset_index(drop=True)
    n = len(df)
    train_idx = int(n * TRAIN_RATIO)
    val_idx = int(n * (TRAIN_RATIO + VAL_RATIO))
    return df.iloc[:train_idx].copy(), df.iloc[train_idx:val_idx].copy(), df.iloc[val_idx:].copy()


def binary_metrics(y_true: pd.Series, y_prob: np.ndarray) -> dict[str, float]:
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-7, 1 - 1e-7)
    metrics = {
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "avg_predicted_rate": float(np.mean(y_prob)),
        "actual_rate": float(np.mean(y_true)),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def strict_json_dumps(data: dict) -> str:
    """Serializes JSON without non-standard NaN/Infinity values."""

    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, tuple):
            return [clean(item) for item in value]
        if isinstance(value, np.generic):
            return clean(value.item())
        if isinstance(value, float):
            return value if np.isfinite(value) else None
        return value

    return json.dumps(clean(data), indent=2, allow_nan=False)


def calibration_table(df: pd.DataFrame, y_prob: np.ndarray) -> list[dict[str, float | int | str]]:
    work = df[[FORFEIT_TARGET_COL]].copy()
    work["pred"] = np.asarray(y_prob, dtype=float)
    bins = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]
    work["bucket"] = pd.cut(work["pred"], bins=bins, include_lowest=True)
    rows = []
    for bucket, group in work.groupby("bucket", observed=False):
        if group.empty:
            continue
        rows.append(
            {
                "bucket": str(bucket),
                "matches": int(len(group)),
                "avg_predicted": float(group["pred"].mean()),
                "actual_rate": float(group[FORFEIT_TARGET_COL].mean()),
                "forfeits": int(group[FORFEIT_TARGET_COL].sum()),
            }
        )
    return rows


def rate_from_group(train_df: pd.DataFrame, key_cols: list[str]) -> dict[tuple, tuple[int, int]]:
    grouped = train_df.groupby(key_cols)[FORFEIT_TARGET_COL].agg(["sum", "count"])
    return {key if isinstance(key, tuple) else (key,): (int(row["sum"]), int(row["count"])) for key, row in grouped.iterrows()}


def baseline_predictions(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> np.ndarray:
    global_success = int(train_df[FORFEIT_TARGET_COL].sum())
    global_count = int(len(train_df))
    global_rate = global_success / global_count
    prior_weight = 20.0

    exact = rate_from_group(train_df, ["is_lan", "region", "organizer"])
    regional = rate_from_group(train_df, ["is_lan", "region"])
    surface = rate_from_group(train_df, ["is_lan"])

    preds = []
    for _, row in eval_df.iterrows():
        keys = [
            (row["is_lan"], row["region"], row["organizer"]),
            (row["is_lan"], row["region"]),
            (row["is_lan"],),
        ]
        lookup = [exact, regional, surface]
        success, count = 0, 0
        for key, table in zip(keys, lookup):
            if key in table:
                success, count = table[key]
                break
        preds.append((success + global_rate * prior_weight) / (count + prior_weight))
    return np.asarray(preds, dtype=float)


def subgroup_metrics(df: pd.DataFrame, y_prob: np.ndarray) -> list[dict[str, float | int | str]]:
    work = df.copy()
    work["pred"] = y_prob
    groups = []

    def add_group(label: str, mask: pd.Series) -> None:
        subset = work[mask]
        if len(subset) < 10:
            return
        metrics = binary_metrics(subset[FORFEIT_TARGET_COL], subset["pred"].to_numpy())
        groups.append({"group": label, "matches": int(len(subset)), **metrics})

    add_group("online", work["is_lan"] == 0)
    add_group("LAN", work["is_lan"] == 1)
    add_group("low-ranked/unranked", (work["max_rank"] > 100) | (work["either_unranked"] == 1))

    for region, count in work["region"].value_counts().head(8).items():
        if count >= 10:
            add_group(f"region: {region}", work["region"] == region)

    for organizer, count in work["organizer"].value_counts().head(10).items():
        if count >= 10:
            add_group(f"event family: {organizer}", work["organizer"] == organizer)

    return groups


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "No rows.\n"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def write_report(
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    best_c: float,
    calibration_method: str,
    val_metrics: dict,
    test_metrics: dict,
    baseline_val_metrics: dict,
    baseline_test_metrics: dict,
    calibration_rows: list[dict],
    subgroup_rows: list[dict],
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Forfeit/default settlement-risk model",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Model",
        "",
        "- Classifier: regularized logistic regression",
        f"- Regularization C: `{best_c}`",
        f"- Calibration: `{calibration_method}` on the validation split",
        "- Sparse team/event/tournament identifiers are represented by leakage-safe smoothed historical rates, not naive one-hot columns.",
        "- Winner model integration is limited to the final Polymarket probability adjustment.",
        "",
        "## Temporal split",
        "",
        markdown_table(
            [
                {
                    "split": "train",
                    "matches": len(train_df),
                    "start": train_df["date"].min().date(),
                    "end": train_df["date"].max().date(),
                    "forfeit_rate": train_df[FORFEIT_TARGET_COL].mean(),
                },
                {
                    "split": "validation",
                    "matches": len(val_df),
                    "start": val_df["date"].min().date(),
                    "end": val_df["date"].max().date(),
                    "forfeit_rate": val_df[FORFEIT_TARGET_COL].mean(),
                },
                {
                    "split": "test",
                    "matches": len(test_df),
                    "start": test_df["date"].min().date(),
                    "end": test_df["date"].max().date(),
                    "forfeit_rate": test_df[FORFEIT_TARGET_COL].mean(),
                },
            ],
            ["split", "matches", "start", "end", "forfeit_rate"],
        ),
        "",
        "## Metrics",
        "",
        markdown_table(
            [
                {"split": "validation model", **val_metrics},
                {"split": "validation baseline", **baseline_val_metrics},
                {"split": "test model", **test_metrics},
                {"split": "test baseline", **baseline_test_metrics},
            ],
            ["split", "log_loss", "brier_score", "roc_auc", "avg_predicted_rate", "actual_rate"],
        ),
        "",
        "## Test calibration buckets",
        "",
        markdown_table(calibration_rows, ["bucket", "matches", "avg_predicted", "actual_rate", "forfeits"]),
        "",
        "## Test subgroup metrics",
        "",
        markdown_table(
            subgroup_rows,
            ["group", "matches", "log_loss", "brier_score", "roc_auc", "avg_predicted_rate", "actual_rate"],
        ),
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def rate_summary(df: pd.DataFrame) -> dict:
    return {
        "matches": int(len(df)),
        "forfeit_matches": int(df[FORFEIT_TARGET_COL].sum()),
        "forfeit_rate": float(df[FORFEIT_TARGET_COL].mean()) if len(df) else 0.0,
    }


def split_summary(name: str, df: pd.DataFrame) -> dict:
    summary = {
        "name": name,
        **rate_summary(df),
        "date_range": {
            "start": str(df["date"].min()) if len(df) else None,
            "end": str(df["date"].max()) if len(df) else None,
        },
    }
    return summary


def grouped_rate_summary(df: pd.DataFrame, group_col: str, limit: int = 12) -> list[dict]:
    rows = []
    grouped = (
        df.groupby(group_col, dropna=False)[FORFEIT_TARGET_COL]
        .agg(["count", "sum", "mean"])
        .sort_values(["count", "mean"], ascending=[False, False])
        .head(limit)
    )
    for value, row in grouped.iterrows():
        rows.append(
            {
                str(group_col): str(value),
                "matches": int(row["count"]),
                "forfeit_matches": int(row["sum"]),
                "forfeit_rate": float(row["mean"]),
            }
        )
    return rows


def build_training_state(
    *,
    base_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metadata: dict,
    calibration_rows: list[dict],
    subgroup_rows: list[dict],
) -> dict:
    online_df = feature_df[feature_df["is_lan"] == 0]
    lan_df = feature_df[feature_df["is_lan"] == 1]
    return {
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_kind": "polymarket_forfeit_default_probability",
        "model_path": str(MODEL_PATH),
        "features_path": str(FEATURES_PATH),
        "report_path": str(REPORT_PATH),
        "metrics_path": str(METRICS_PATH),
        "source": {
            "raw_matches": int(len(base_df)),
            "feature_rows": int(len(feature_df)),
            "date_range": {
                "start": str(feature_df["date"].min()) if len(feature_df) else None,
                "end": str(feature_df["date"].max()) if len(feature_df) else None,
            },
        },
        "target": {
            "column": FORFEIT_TARGET_COL,
            "definition": "1 when a match has a full default, forfeit, disqualification, walkover, or partial default that can affect Polymarket settlement; otherwise 0.",
            "overall": rate_summary(feature_df),
            "online": rate_summary(online_df),
            "lan": rate_summary(lan_df),
        },
        "splits": {
            "train": split_summary("train", train_df),
            "validation": split_summary("validation", val_df),
            "test": split_summary("test", test_df),
        },
        "model": {
            "type": metadata["model_type"],
            "calibration_method": metadata["calibration_method"],
            "best_c": metadata["best_c"],
            "validation_scores": metadata["validation_scores"],
            "feature_count": len(metadata["features"]),
            "numeric_feature_count": len(metadata["numeric_features"]),
            "categorical_feature_count": len(metadata["categorical_features"]),
            "features": metadata["features"],
        },
        "metrics": metadata["metrics"],
        "calibration_buckets": calibration_rows,
        "subgroups": subgroup_rows,
        "breakdowns": {
            "format": grouped_rate_summary(feature_df, "format"),
            "region": grouped_rate_summary(feature_df, "region"),
            "event_family": grouped_rate_summary(feature_df, "organizer"),
        },
    }


def train_forfeit_model(calibration_method: str = "sigmoid") -> dict:
    base_df = build_forfeit_base_rows()
    feature_df = build_forfeit_feature_frame(base_df)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_parquet(FEATURES_PATH, index=False)

    train_df, val_df, test_df = split_temporally(feature_df)
    X_train = train_df[FORFEIT_MODEL_FEATURES]
    y_train = train_df[FORFEIT_TARGET_COL]
    X_val = val_df[FORFEIT_MODEL_FEATURES]
    y_val = val_df[FORFEIT_TARGET_COL]
    X_test = test_df[FORFEIT_MODEL_FEATURES]
    y_test = test_df[FORFEIT_TARGET_COL]

    candidates = [0.03, 0.1, 0.3, 1.0, 3.0]
    val_scores = []
    best_c = candidates[0]
    best_pipeline = None
    best_score = float("inf")
    for c in candidates:
        pipeline = build_pipeline(c)
        pipeline.fit(X_train, y_train)
        val_prob = pipeline.predict_proba(X_val)[:, 1]
        score = log_loss(y_val, np.clip(val_prob, 1e-7, 1 - 1e-7))
        val_scores.append({"C": c, "validation_log_loss": float(score)})
        if score < best_score:
            best_score = score
            best_c = c
            best_pipeline = pipeline

    calibrated_model = fit_prefit_calibrator(best_pipeline, X_val, y_val, calibration_method)
    val_prob = calibrated_model.predict_proba(X_val)[:, 1]
    test_prob = calibrated_model.predict_proba(X_test)[:, 1]
    baseline_val_prob = baseline_predictions(train_df, val_df)
    baseline_test_prob = baseline_predictions(train_df, test_df)

    val_metrics = binary_metrics(y_val, val_prob)
    test_metrics = binary_metrics(y_test, test_prob)
    baseline_val_metrics = binary_metrics(y_val, baseline_val_prob)
    baseline_test_metrics = binary_metrics(y_test, baseline_test_prob)
    calibration_rows = calibration_table(test_df, test_prob)
    subgroup_rows = subgroup_metrics(test_df, test_prob)

    history_state = build_full_history_state(base_df)
    metadata = {
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": "regularized_logistic_regression",
        "calibration_method": calibration_method,
        "best_c": best_c,
        "features": FORFEIT_MODEL_FEATURES,
        "numeric_features": NUMERIC_FORFEIT_FEATURES,
        "categorical_features": CATEGORICAL_FORFEIT_FEATURES,
        "validation_scores": val_scores,
        "target": FORFEIT_TARGET_COL,
        "split": {
            "train": [str(train_df["date"].min()), str(train_df["date"].max()), int(len(train_df))],
            "validation": [str(val_df["date"].min()), str(val_df["date"].max()), int(len(val_df))],
            "test": [str(test_df["date"].min()), str(test_df["date"].max()), int(len(test_df))],
        },
        "metrics": {
            "validation": val_metrics,
            "test": test_metrics,
            "baseline_validation": baseline_val_metrics,
            "baseline_test": baseline_test_metrics,
        },
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": calibrated_model,
            "history_state": history_state,
            "metadata": metadata,
        },
        MODEL_PATH,
    )

    write_report(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        best_c=best_c,
        calibration_method=calibration_method,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        baseline_val_metrics=baseline_val_metrics,
        baseline_test_metrics=baseline_test_metrics,
        calibration_rows=calibration_rows,
        subgroup_rows=subgroup_rows,
    )

    METRICS_PATH.write_text(
        strict_json_dumps(
            {
                "metadata": metadata,
                "calibration_buckets": calibration_rows,
                "subgroups": subgroup_rows,
            }
        ),
        encoding="utf-8",
    )

    training_state = build_training_state(
        base_df=base_df,
        feature_df=feature_df,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        metadata=metadata,
        calibration_rows=calibration_rows,
        subgroup_rows=subgroup_rows,
    )
    STATE_PATH.write_text(strict_json_dumps(training_state), encoding="utf-8")

    return {
        "model_path": str(MODEL_PATH),
        "features_path": str(FEATURES_PATH),
        "report_path": str(REPORT_PATH),
        "metrics_path": str(METRICS_PATH),
        "state_path": str(STATE_PATH),
        "metadata": metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Polymarket settlement forfeit/default model.")
    parser.add_argument(
        "--calibration-method",
        choices=["sigmoid", "isotonic"],
        default="sigmoid",
        help="Probability calibration method fitted on the temporal validation split.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = train_forfeit_model(calibration_method=args.calibration_method)
    print(f"Forfeit model saved to {result['model_path']}")
    print(f"Forfeit training state saved to {result['state_path']}")
    print(f"Evaluation report saved to {result['report_path']}")
