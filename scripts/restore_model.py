"""Back up and restore trained winner model checkpoints by version ID.

Usage:
    python scripts/restore_model.py --list
    python scripts/restore_model.py --backup-only
    python scripts/restore_model.py <version_id>
"""
import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from evaluation import shadow_ledger


LIVE_MODEL_FILENAME = "best_mvp_model.pt"
LIVE_SCALER_FILENAME = "scaler.pkl"
TRAINING_STATE_FILENAME = "training_state.json"


def _resolve_archive_paths(version_row: dict, checkpoint_dir: Path) -> tuple[Path, Path]:
    version_id = version_row["version_id"]
    fallback_dir = checkpoint_dir / "archive" / version_id
    weights_raw = version_row.get("weights_path")
    scaler_raw = version_row.get("scaler_path")

    weights_path = Path(weights_raw) if weights_raw else fallback_dir / LIVE_MODEL_FILENAME
    scaler_path = Path(scaler_raw) if scaler_raw else fallback_dir / LIVE_SCALER_FILENAME

    if not weights_path.is_absolute():
        weights_path = PROJECT_ROOT / weights_path
    if not scaler_path.is_absolute():
        scaler_path = PROJECT_ROOT / scaler_path

    if not weights_path.exists() and (fallback_dir / LIVE_MODEL_FILENAME).exists():
        weights_path = fallback_dir / LIVE_MODEL_FILENAME
    if not scaler_path.exists() and (fallback_dir / LIVE_SCALER_FILENAME).exists():
        scaler_path = fallback_dir / LIVE_SCALER_FILENAME
    return weights_path, scaler_path


def backup_live_artifacts(
    *,
    label: str = "manual",
    data_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
) -> Path | None:
    """Copy the current live checkpoint, scaler, and training_state.json into a backup folder."""
    data_dir = Path(data_dir or config.DATA_DIR)
    checkpoint_dir = Path(checkpoint_dir or config.CHECKPOINT_DIR)

    sources = [
        checkpoint_dir / LIVE_MODEL_FILENAME,
        checkpoint_dir / LIVE_SCALER_FILENAME,
        data_dir / TRAINING_STATE_FILENAME,
    ]
    existing = [src for src in sources if src.exists()]
    if not existing:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backups_root = checkpoint_dir / "backups"
    backup_dir = backups_root / f"{label}_{timestamp}"
    suffix = 1
    while backup_dir.exists():
        backup_dir = backups_root / f"{label}_{timestamp}_{suffix}"
        suffix += 1

    backup_dir.mkdir(parents=True, exist_ok=True)
    for src in existing:
        shutil.copy2(src, backup_dir / src.name)
    return backup_dir


def list_versions(checkpoint_dir: Path | None = None) -> list[dict]:
    """Return registered model versions with archive status and active flag."""
    checkpoint_dir = Path(checkpoint_dir or config.CHECKPOINT_DIR)
    active_id = shadow_ledger.get_latest_version_id()

    conn = shadow_ledger.get_db()
    try:
        rows = conn.execute(
            """SELECT version_id, trained_at, test_brier_score, test_log_loss,
                      weights_path, scaler_path, data_stats_json
               FROM model_versions
               ORDER BY trained_at DESC"""
        ).fetchall()
    finally:
        conn.close()

    versions = []
    for row in rows:
        record = dict(row)
        weights_path, scaler_path = _resolve_archive_paths(record, checkpoint_dir)
        stats = {}
        if record.get("data_stats_json"):
            try:
                stats = json.loads(record["data_stats_json"]) or {}
            except json.JSONDecodeError:
                stats = {}
        versions.append({
            "version_id": record["version_id"],
            "trained_at": record["trained_at"],
            "active": record["version_id"] == active_id,
            "brier": record.get("test_brier_score"),
            "log_loss": record.get("test_log_loss"),
            "training_maps": stats.get("total_maps"),
            "archive_ready": weights_path.exists() and scaler_path.exists(),
        })
    return versions


def restore_version(
    version_id: str,
    *,
    backup_current: bool = True,
    data_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
) -> dict:
    """Restore an archived model version without modifying processed parquet files."""
    data_dir = Path(data_dir or config.DATA_DIR)
    checkpoint_dir = Path(checkpoint_dir or config.CHECKPOINT_DIR)

    conn = shadow_ledger.get_db()
    try:
        row = conn.execute(
            "SELECT * FROM model_versions WHERE version_id = ?",
            (version_id,),
        ).fetchone()
    finally:
        conn.close()

    if not row:
        raise ValueError(f"Model version '{version_id}' was not found in shadow_ledger.db.")

    record = dict(row)
    weights_src, scaler_src = _resolve_archive_paths(record, checkpoint_dir)
    if not weights_src.exists():
        raise FileNotFoundError(f"Archived model weights not found: {weights_src}")
    if not scaler_src.exists():
        raise FileNotFoundError(f"Archived scaler not found: {scaler_src}")

    backup_dir = None
    if backup_current:
        backup_dir = backup_live_artifacts(
            label="pre_restore",
            data_dir=data_dir,
            checkpoint_dir=checkpoint_dir,
        )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    weights_dst = checkpoint_dir / LIVE_MODEL_FILENAME
    scaler_dst = checkpoint_dir / LIVE_SCALER_FILENAME
    shutil.copy2(weights_src, weights_dst)
    shutil.copy2(scaler_src, scaler_dst)

    state_restored = False
    raw_stats = record.get("data_stats_json")
    if raw_stats:
        try:
            parsed_stats = json.loads(raw_stats)
        except json.JSONDecodeError:
            parsed_stats = None
        if isinstance(parsed_stats, dict) and parsed_stats:
            state_path = data_dir / TRAINING_STATE_FILENAME
            state_path.write_text(json.dumps(parsed_stats, indent=4), encoding="utf-8")
            state_restored = True

    shadow_ledger.set_active_model_version(version_id)
    return {
        "version_id": version_id,
        "backup_dir": backup_dir,
        "weights_dst": weights_dst,
        "scaler_dst": scaler_dst,
        "training_state_restored": state_restored,
    }


def _format_metric(value: float | None) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "-"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Back up and restore trained winner model checkpoints."
    )
    parser.add_argument("version_id", nargs="?", help="Archived version_id to restore.")
    parser.add_argument("--list", action="store_true", help="List registered versions and exit.")
    parser.add_argument(
        "--backup-only",
        action="store_true",
        help="Back up the current live checkpoint, scaler, and training_state.json.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip backing up the current live files before restoring.",
    )
    args = parser.parse_args(argv)

    if args.list:
        versions = list_versions()
        if not versions:
            print("No model versions found in shadow_ledger.db.")
            return 0
        print(f"{'ACTIVE':<6} {'VERSION_ID':<20} {'TRAINED_AT':<19} {'BRIER':>7} {'LOGLOSS':>7} {'MAPS':>7} {'ARCHIVE':<7}")
        for item in versions:
            active_mark = "*" if item["active"] else ""
            archive_mark = "ready" if item["archive_ready"] else "missing"
            maps = str(item["training_maps"]) if item["training_maps"] is not None else "-"
            print(
                f"{active_mark:<6} {item['version_id']:<20} {item['trained_at']:<19} "
                f"{_format_metric(item['brier']):>7} {_format_metric(item['log_loss']):>7} "
                f"{maps:>7} {archive_mark:<7}"
            )
        return 0

    if args.backup_only:
        backup_dir = backup_live_artifacts(label="manual")
        if backup_dir is None:
            print("No live checkpoint, scaler, or training_state.json found to back up.")
            return 1
        print(f"Backed up live model artifacts to {backup_dir}")
        return 0

    if not args.version_id:
        parser.print_help()
        return 2

    result = restore_version(args.version_id, backup_current=not args.no_backup)
    if result["backup_dir"] is not None:
        print(f"Backed up previous live artifacts to {result['backup_dir']}")
    print(f"Restored {result['version_id']} -> {result['weights_dst']} and {result['scaler_dst']}")
    if result["training_state_restored"]:
        print("Restored data/training_state.json from version metadata.")
    else:
        print("Note: version metadata had no data_stats_json; training_state.json left unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
