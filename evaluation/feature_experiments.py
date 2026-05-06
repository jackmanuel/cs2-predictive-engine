"""
Walk-forward feature experiments for the CS2 map predictor.

This module trains the same small neural network across controlled feature
variants, seeds, and temporal validation folds. It is designed for qualitative
model development: answer whether a feature tweak improves future-map
performance before promoting it into the production training pipeline.
"""

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from config import (
    BATCH_SIZE,
    DROPOUT_RATE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    LEARNING_RATE,
    PROCESSED_DIR,
    PROJECT_ROOT,
)
from evaluation.metrics import compute_metrics
from model.net import MatchPredictor
from processing.features import MODEL_FEATURES, TARGET_COL

LOGGER = logging.getLogger(__name__)

DEFAULT_FEATURES_PATH = PROCESSED_DIR / "features.parquet"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"

H2H_METADATA_COLUMNS = ["date", "match_id", "team_a_id", "team_b_id", TARGET_COL]


@dataclass(frozen=True)
class FeatureVariant:
    name: str
    features: tuple[str, ...]
    description: str
    h2h_window_days: int | None = None


@dataclass(frozen=True)
class TemporalFold:
    name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    eval_start: pd.Timestamp
    eval_end: pd.Timestamp
    train_rows: int
    eval_rows: int
    eval_matches: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run walk-forward feature ablation experiments."
    )
    parser.add_argument(
        "--features-path",
        default=str(DEFAULT_FEATURES_PATH),
        help="Processed feature parquet to use as the experiment source.",
    )
    parser.add_argument(
        "--preset",
        choices=["promising", "full"],
        default="promising",
        help="Variant set to evaluate. Default: promising.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Explicit variant names to run instead of the selected preset.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=4,
        help="Number of trailing walk-forward folds. Default: 4.",
    )
    parser.add_argument(
        "--fold-days",
        type=int,
        default=7,
        help="Number of calendar days in each outer evaluation fold. Default: 7.",
    )
    parser.add_argument(
        "--min-train-days",
        type=int,
        default=60,
        help="Minimum history required before a fold can be used. Default: 60.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        help="Random seeds to run for every variant/fold. Default: 1 2 3 4 5.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Maximum epochs per run. Default: {EPOCHS}.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=EARLY_STOPPING_PATIENCE,
        help=f"Early stopping patience. Default: {EARLY_STOPPING_PATIENCE}.",
    )
    parser.add_argument(
        "--inner-val-ratio",
        type=float,
        default=0.15,
        help="Temporal validation fraction carved from each fold's training data.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size. Default: {BATCH_SIZE}.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
        help=f"Adam learning rate. Default: {LEARNING_RATE}.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Adam weight decay. Default: 1e-4.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help=(
            "Optional report name stem. Defaults to a timestamped name derived "
            "from the preset/folds/seeds."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Detailed per-run CSV output path. Defaults to a timestamped report.",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Aggregated summary CSV output path. Defaults to a timestamped report.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print folds and variants without training.",
    )
    return parser.parse_args()


def build_variant_catalog() -> dict[str, FeatureVariant]:
    baseline = tuple(MODEL_FEATURES)

    def without(*columns: str) -> tuple[str, ...]:
        remove = set(columns)
        return tuple(feature for feature in baseline if feature not in remove)

    def with_h2h_window(days: int, base_features: Iterable[str] = baseline) -> tuple[str, ...]:
        features = []
        for feature in base_features:
            if feature.startswith("h2h_a_wins"):
                features.append(f"h2h_a_wins_{days}d")
            elif feature.startswith("h2h_b_wins"):
                features.append(f"h2h_b_wins_{days}d")
            else:
                features.append(feature)
        return tuple(features)

    def without_h2h(base_features: Iterable[str] = baseline) -> tuple[str, ...]:
        return tuple(feature for feature in base_features if not feature.startswith("h2h_"))

    no_lan_dom_res = without("lan_rate_diff", "dominance_diff", "resilience_diff")

    variants = [
        FeatureVariant(
            "baseline_all",
            baseline,
            "Current production feature set.",
            h2h_window_days=30,
        ),
        FeatureVariant(
            "no_lan_rate",
            without("lan_rate_diff"),
            "Remove historical LAN-rate differential.",
        ),
        FeatureVariant(
            "no_dominance_resilience",
            without("dominance_diff", "resilience_diff"),
            "Remove raw round-margin dominance and resilience features.",
        ),
        FeatureVariant(
            "no_lan_dom_resilience",
            no_lan_dom_res,
            "Remove LAN rate, dominance, and resilience together.",
        ),
        FeatureVariant(
            "h2h_14d_counts",
            with_h2h_window(14),
            "Replace production H2H counts with 14-day counts.",
            h2h_window_days=14,
        ),
        FeatureVariant(
            "h2h_30d_counts",
            with_h2h_window(30),
            "Use 30-day H2H map counts.",
            h2h_window_days=30,
        ),
        FeatureVariant(
            "h2h_90d_counts",
            with_h2h_window(90),
            "Replace all-time prior H2H map counts with 90-day counts.",
            h2h_window_days=90,
        ),
        FeatureVariant(
            "no_lan_dom_res_h2h_14d",
            with_h2h_window(14, no_lan_dom_res),
            "Remove the most suspicious features and use 14-day H2H counts.",
            h2h_window_days=14,
        ),
        FeatureVariant(
            "no_lan_dom_res_h2h_30d",
            with_h2h_window(30, no_lan_dom_res),
            "Remove the most suspicious features and use 30-day H2H counts.",
            h2h_window_days=30,
        ),
        FeatureVariant(
            "no_lan_dom_res_h2h_90d",
            with_h2h_window(90, no_lan_dom_res),
            "Remove the most suspicious features and use 90-day H2H counts.",
            h2h_window_days=90,
        ),
        FeatureVariant(
            "no_7d_form",
            without("win_rate_7d_diff"),
            "Remove short-window form, which may be noisy.",
        ),
        FeatureVariant(
            "single_sos_90d",
            without("sos_diff"),
            "Keep 90-day strength-of-schedule only.",
        ),
        FeatureVariant(
            "single_sos_30d",
            without("sos_90d_diff"),
            "Keep 30-day strength-of-schedule only.",
        ),
        FeatureVariant(
            "no_h2h",
            without_h2h(),
            "Remove H2H counts entirely.",
        ),
    ]
    return {variant.name: variant for variant in variants}


def preset_names(preset: str) -> list[str]:
    promising = [
        "baseline_all",
        "no_lan_rate",
        "no_dominance_resilience",
        "no_lan_dom_resilience",
        "h2h_14d_counts",
        "h2h_90d_counts",
        "no_lan_dom_res_h2h_14d",
        "no_lan_dom_res_h2h_90d",
    ]
    if preset == "promising":
        return promising
    return [
        *promising,
        "no_7d_form",
        "single_sos_90d",
        "single_sos_30d",
        "no_h2h",
    ]


def load_feature_frame(features_path: Path) -> pd.DataFrame:
    if not features_path.exists():
        raise FileNotFoundError(f"Feature parquet not found: {features_path}")

    df = pd.read_parquet(features_path)
    required = sorted(
        set(H2H_METADATA_COLUMNS)
        | {feature for feature in MODEL_FEATURES if not feature.startswith("h2h_")}
    )
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise RuntimeError(f"Feature parquet is missing required columns: {missing}")

    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


def add_h2h_window_features(df: pd.DataFrame, window_days: int) -> pd.DataFrame:
    a_col = f"h2h_a_wins_{window_days}d"
    b_col = f"h2h_b_wins_{window_days}d"
    if a_col in df.columns and b_col in df.columns:
        return df

    enriched = df.copy()
    a_values = np.zeros(len(enriched), dtype=np.float32)
    b_values = np.zeros(len(enriched), dtype=np.float32)
    histories: dict[tuple[str, str], list[tuple[pd.Timestamp, str]]] = {}
    window = pd.Timedelta(days=window_days)

    for idx, row in enriched.iterrows():
        team_a = str(row["team_a_id"])
        team_b = str(row["team_b_id"])
        pair = tuple(sorted((team_a, team_b)))
        current_date = row["date"]
        cutoff = current_date - window
        history = histories.setdefault(pair, [])
        recent = [(date, winner) for date, winner in history if date >= cutoff]
        histories[pair] = recent

        a_values[idx] = sum(1 for _, winner in recent if winner == team_a)
        b_values[idx] = sum(1 for _, winner in recent if winner == team_b)

        winner = team_a if int(row[TARGET_COL]) == 1 else team_b
        recent.append((current_date, winner))

    enriched[a_col] = a_values
    enriched[b_col] = b_values
    return enriched


def ensure_variant_columns(df: pd.DataFrame, variants: Iterable[FeatureVariant]) -> pd.DataFrame:
    enriched = df
    for variant in variants:
        if variant.h2h_window_days is not None:
            enriched = add_h2h_window_features(enriched, variant.h2h_window_days)

    required = sorted({feature for variant in variants for feature in variant.features})
    missing = [feature for feature in required if feature not in enriched.columns]
    if missing:
        raise RuntimeError(f"Experiment features are missing from the frame: {missing}")
    return enriched


def build_temporal_folds(
    df: pd.DataFrame,
    folds: int,
    fold_days: int,
    min_train_days: int,
) -> list[TemporalFold]:
    if folds < 1:
        raise ValueError("--folds must be at least 1.")
    if fold_days < 1:
        raise ValueError("--fold-days must be at least 1.")

    first_day = df["date"].min().floor("D")
    latest_end = df["date"].max().floor("D") + pd.Timedelta(days=1)
    temporal_folds = []

    for index in range(folds):
        eval_end = latest_end - pd.Timedelta(days=fold_days * (folds - index - 1))
        eval_start = eval_end - pd.Timedelta(days=fold_days)
        train_end = eval_start

        if train_end - first_day < pd.Timedelta(days=min_train_days):
            LOGGER.warning(
                "Skipping fold ending %s because it has less than %s days of training history.",
                eval_end.date(),
                min_train_days,
            )
            continue

        train_mask = df["date"] < train_end
        eval_mask = (df["date"] >= eval_start) & (df["date"] < eval_end)
        train_rows = int(train_mask.sum())
        eval_rows = int(eval_mask.sum())
        if train_rows == 0 or eval_rows == 0:
            LOGGER.warning(
                "Skipping fold %s because train_rows=%s eval_rows=%s.",
                index + 1,
                train_rows,
                eval_rows,
            )
            continue

        temporal_folds.append(
            TemporalFold(
                name=f"fold_{len(temporal_folds) + 1}",
                train_start=df.loc[train_mask, "date"].min(),
                train_end=train_end,
                eval_start=eval_start,
                eval_end=eval_end,
                train_rows=train_rows,
                eval_rows=eval_rows,
                eval_matches=df.loc[eval_mask, "match_id"].nunique(),
            )
        )

    if not temporal_folds:
        raise RuntimeError("No usable temporal folds were produced.")
    return temporal_folds


def mirror_for_features(df: pd.DataFrame) -> pd.DataFrame:
    mirrored = df.copy()

    for column in df.columns:
        if column.endswith("_diff"):
            mirrored[column] = -df[column]

    for column in df.columns:
        if column.startswith("team_a_"):
            partner = "team_b_" + column[len("team_a_") :]
            if partner in df.columns:
                mirrored[column] = df[partner]
                mirrored[partner] = df[column]

    for column in df.columns:
        if column.startswith("h2h_a_wins"):
            partner = "h2h_b_wins" + column[len("h2h_a_wins") :]
            if partner in df.columns:
                mirrored[column] = df[partner]
                mirrored[partner] = df[column]

    mirrored[TARGET_COL] = 1 - df[TARGET_COL]
    return pd.concat([df, mirrored], ignore_index=True)


def make_tensor_dataset(
    df: pd.DataFrame,
    feature_columns: tuple[str, ...],
    scaler: StandardScaler,
    fit_scaler: bool,
) -> TensorDataset:
    x_raw = df.loc[:, feature_columns].values.astype(np.float32)
    y_raw = df[TARGET_COL].values.astype(np.float32).reshape(-1, 1)
    if fit_scaler:
        x_values = scaler.fit_transform(x_raw)
    else:
        x_values = scaler.transform(x_raw)

    x_tensor = torch.tensor(x_values, dtype=torch.float32)
    y_tensor = torch.tensor(y_raw, dtype=torch.float32)
    return TensorDataset(x_tensor, y_tensor)


def split_inner_train_val(
    train_df: pd.DataFrame,
    inner_val_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < inner_val_ratio < 0.5:
        raise ValueError("--inner-val-ratio must be between 0 and 0.5.")

    split_idx = int(len(train_df) * (1 - inner_val_ratio))
    if split_idx <= 0 or split_idx >= len(train_df):
        raise RuntimeError("Inner train/validation split produced an empty partition.")
    return train_df.iloc[:split_idx].copy(), train_df.iloc[split_idx:].copy()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_and_score(
    df: pd.DataFrame,
    variant: FeatureVariant,
    fold: TemporalFold,
    seed: int,
    args: argparse.Namespace,
) -> dict:
    set_seed(seed)

    train_df = df[df["date"] < fold.train_end].copy()
    eval_df = df[(df["date"] >= fold.eval_start) & (df["date"] < fold.eval_end)].copy()
    inner_train_df, inner_val_df = split_inner_train_val(train_df, args.inner_val_ratio)
    inner_train_df = mirror_for_features(inner_train_df)

    scaler = StandardScaler()
    train_dataset = make_tensor_dataset(inner_train_df, variant.features, scaler, fit_scaler=True)
    val_dataset = make_tensor_dataset(inner_val_df, variant.features, scaler, fit_scaler=False)
    eval_dataset = make_tensor_dataset(eval_df, variant.features, scaler, fit_scaler=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = MatchPredictor(input_dim=len(variant.features))
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    best_state = None
    best_val_loss = float("inf")
    patience_counter = 0
    epochs_run = 0

    for epoch in range(args.epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                preds = model(x_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * len(x_batch)
        val_loss /= len(val_dataset)
        scheduler.step(val_loss)

        epochs_run = epoch + 1
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break

    if best_state is None:
        raise RuntimeError(f"No model state was captured for {variant.name} {fold.name} seed {seed}.")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        x_eval, y_eval = eval_dataset.tensors
        preds = model(x_eval).numpy().reshape(-1)
        y_true = y_eval.numpy().reshape(-1)

    metrics = compute_metrics(y_true, preds)
    return {
        "variant": variant.name,
        "description": variant.description,
        "fold": fold.name,
        "seed": seed,
        "num_features": len(variant.features),
        "features": json.dumps(list(variant.features)),
        "h2h_window_days": variant.h2h_window_days,
        "train_start": fold.train_start.isoformat(),
        "train_end": fold.train_end.isoformat(),
        "eval_start": fold.eval_start.isoformat(),
        "eval_end": fold.eval_end.isoformat(),
        "train_rows": fold.train_rows,
        "eval_rows": fold.eval_rows,
        "eval_matches": fold.eval_matches,
        "inner_train_rows_after_mirror": len(inner_train_df),
        "inner_val_rows": len(inner_val_df),
        "epochs_run": epochs_run,
        "best_inner_val_loss": best_val_loss,
        "brier_score": metrics["brier_score"],
        "log_loss": metrics["log_loss"],
        "accuracy": metrics["accuracy"],
        "roc_auc": metrics["roc_auc"],
    }


def summarize_results(results_df: pd.DataFrame) -> pd.DataFrame:
    grouped = results_df.groupby("variant", sort=False)
    summary = grouped.agg(
        runs=("variant", "size"),
        num_features=("num_features", "first"),
        mean_brier=("brier_score", "mean"),
        std_brier=("brier_score", "std"),
        mean_log_loss=("log_loss", "mean"),
        std_log_loss=("log_loss", "std"),
        mean_accuracy=("accuracy", "mean"),
        mean_roc_auc=("roc_auc", "mean"),
        mean_epochs=("epochs_run", "mean"),
        description=("description", "first"),
    ).reset_index()

    baseline = results_df[results_df["variant"] == "baseline_all"][
        ["fold", "seed", "brier_score", "log_loss"]
    ].rename(
        columns={
            "brier_score": "baseline_brier",
            "log_loss": "baseline_log_loss",
        }
    )
    deltas = results_df.merge(baseline, on=["fold", "seed"], how="left")
    deltas["delta_brier_vs_baseline"] = deltas["brier_score"] - deltas["baseline_brier"]
    deltas["delta_log_loss_vs_baseline"] = deltas["log_loss"] - deltas["baseline_log_loss"]
    deltas["brier_win_vs_baseline"] = deltas["delta_brier_vs_baseline"] < 0
    deltas["log_loss_win_vs_baseline"] = deltas["delta_log_loss_vs_baseline"] < 0

    delta_summary = deltas.groupby("variant", sort=False).agg(
        mean_delta_brier_vs_baseline=("delta_brier_vs_baseline", "mean"),
        mean_delta_log_loss_vs_baseline=("delta_log_loss_vs_baseline", "mean"),
        brier_wins_vs_baseline=("brier_win_vs_baseline", "sum"),
        log_loss_wins_vs_baseline=("log_loss_win_vs_baseline", "sum"),
    ).reset_index()

    summary = summary.merge(delta_summary, on="variant", how="left")
    return summary.sort_values(["mean_brier", "mean_log_loss"]).reset_index(drop=True)


def write_outputs(
    results: list[dict],
    output_path: Path,
    summary_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    summary_df = summarize_results(results_df)
    summary_df.to_csv(summary_path, index=False)
    return results_df, summary_df


def slugify(value: str) -> str:
    cleaned = []
    for char in value.lower():
        if char.isalnum():
            cleaned.append(char)
        elif char in {"-", "_"}:
            cleaned.append(char)
        else:
            cleaned.append("_")
    slug = "".join(cleaned).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "feature_experiment"


def default_report_stem(args: argparse.Namespace) -> str:
    if args.run_name:
        return slugify(args.run_name)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    variant_label = "custom" if args.variants else args.preset
    return slugify(
        f"feature_experiment_{variant_label}_{args.fold_days}d_"
        f"{args.folds}folds_{len(args.seeds)}seeds_{timestamp}"
    )


def resolve_output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    stem = default_report_stem(args)
    output_path = Path(args.output) if args.output else DEFAULT_REPORT_DIR / f"{stem}_results.csv"
    summary_path = (
        Path(args.summary_output)
        if args.summary_output
        else DEFAULT_REPORT_DIR / f"{stem}_summary.csv"
    )
    return output_path, summary_path


def print_plan(variants: list[FeatureVariant], folds: list[TemporalFold], seeds: list[int]) -> None:
    print("\n=== Feature Experiment Plan ===")
    print(f"Variants: {len(variants)}")
    for variant in variants:
        print(f"  - {variant.name}: {len(variant.features)} features")
    print(f"Folds: {len(folds)}")
    for fold in folds:
        print(
            f"  - {fold.name}: train < {fold.train_end.date()} | "
            f"eval {fold.eval_start.date()} to {(fold.eval_end - pd.Timedelta(days=1)).date()} "
            f"({fold.eval_matches} matches, {fold.eval_rows} maps)"
        )
    print(f"Seeds: {seeds}")
    print(f"Total training runs: {len(variants) * len(folds) * len(seeds)}")


def print_summary(summary_df: pd.DataFrame, output_path: Path, summary_path: Path) -> None:
    display_columns = [
        "variant",
        "runs",
        "num_features",
        "mean_brier",
        "mean_log_loss",
        "mean_accuracy",
        "mean_roc_auc",
        "mean_delta_brier_vs_baseline",
        "brier_wins_vs_baseline",
    ]
    display = summary_df[display_columns].copy()
    for column in [
        "mean_brier",
        "mean_log_loss",
        "mean_accuracy",
        "mean_roc_auc",
        "mean_delta_brier_vs_baseline",
    ]:
        display[column] = display[column].map(lambda value: f"{value:.5f}")

    print("\n=== Feature Experiment Summary ===")
    print(display.to_string(index=False))
    print(f"\nSaved detailed results: {output_path}")
    print(f"Saved summary: {summary_path}")


def select_variants(args: argparse.Namespace) -> list[FeatureVariant]:
    catalog = build_variant_catalog()
    names = args.variants if args.variants else preset_names(args.preset)
    unknown = [name for name in names if name not in catalog]
    if unknown:
        raise ValueError(
            f"Unknown variants: {unknown}. Available variants: {sorted(catalog)}"
        )
    return [catalog[name] for name in names]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    variants = select_variants(args)
    feature_df = load_feature_frame(Path(args.features_path))
    feature_df = ensure_variant_columns(feature_df, variants)
    folds = build_temporal_folds(
        feature_df,
        folds=args.folds,
        fold_days=args.fold_days,
        min_train_days=args.min_train_days,
    )

    print_plan(variants, folds, args.seeds)
    if args.dry_run:
        return 0

    total_runs = len(variants) * len(folds) * len(args.seeds)
    results = []
    run_number = 0

    for variant in variants:
        for fold in folds:
            for seed in args.seeds:
                run_number += 1
                LOGGER.info(
                    "[%s/%s] variant=%s fold=%s seed=%s",
                    run_number,
                    total_runs,
                    variant.name,
                    fold.name,
                    seed,
                )
                results.append(train_and_score(feature_df, variant, fold, seed, args))

    output_path, summary_path = resolve_output_paths(args)
    _, summary_df = write_outputs(results, output_path, summary_path)
    print_summary(summary_df, output_path, summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
