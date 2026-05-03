import argparse
import json
import random
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
STALE_MAP_THRESHOLD = 500
HLTV_MATCHES_FILE = DATA_DIR / "raw" / "hltv_matches.json"
TRAINING_STATE_FILE = DATA_DIR / "training_state.json"
SHADOW_LEDGER_DB = DATA_DIR / "shadow_ledger.db"
DEFAULT_INTERVAL_HOURS = 2.0
DEFAULT_JITTER_MINUTES = 30.0


def run_step(label: str, command: list[str]) -> None:
    print(f"\n=== {label} ===")
    print(" ".join(command))
    subprocess.run(command, check=True)


def load_scraped_map_count() -> int | None:
    if not HLTV_MATCHES_FILE.exists():
        return None

    try:
        from processing.clean import build_clean_maps
    except ImportError as exc:
        print(f"WARNING: Could not import cleaning helpers for the canonical scrape count: {exc}")
        return None

    try:
        clean_maps = build_clean_maps()
    except Exception as exc:
        print(f"WARNING: Could not count cleaned canonical maps from {HLTV_MATCHES_FILE}: {exc}")
        return None

    return len(clean_maps)


def load_training_state_map_count() -> int | None:
    if not TRAINING_STATE_FILE.exists():
        return None

    try:
        with open(TRAINING_STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARNING: Could not read training state at {TRAINING_STATE_FILE}: {exc}")
        return None

    return state.get("total_maps")


def load_latest_ledger_training_maps() -> tuple[str | None, int | None]:
    db_path = Path(SHADOW_LEDGER_DB)
    if not db_path.exists():
        return None, None

    try:
        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute(
                """
                SELECT version_id, data_stats_json
                FROM model_versions
                ORDER BY trained_at DESC
                LIMIT 1
                """
            ).fetchone()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        print(f"WARNING: Could not read shadow ledger versions from {db_path}: {exc}")
        return None, None

    if not row:
        return None, None

    version_id, data_stats_json = row
    try:
        data_stats = json.loads(data_stats_json or "{}")
    except json.JSONDecodeError:
        print(f"WARNING: Latest shadow ledger version {version_id} has invalid data_stats_json.")
        return version_id, None

    return version_id, data_stats.get("total_maps")


def warn_if_model_is_stale() -> None:
    scraped_maps = load_scraped_map_count()
    training_state_maps = load_training_state_map_count()
    ledger_version_id, ledger_maps = load_latest_ledger_training_maps()

    if scraped_maps is None:
        print("WARNING: Could not determine cleaned canonical map count; skipping model freshness check.")
        return

    model_maps = training_state_maps if training_state_maps is not None else ledger_maps
    if model_maps is None:
        print("WARNING: Could not determine model training map count; skipping model freshness check.")
        return

    maps_behind = scraped_maps - model_maps
    print("\n=== Model freshness ===")
    print(f"Cleaned canonical maps: {scraped_maps}")
    print(f"Training state maps:    {training_state_maps if training_state_maps is not None else 'unknown'}")
    if ledger_version_id:
        print(f"Ledger version maps:    {ledger_maps if ledger_maps is not None else 'unknown'} ({ledger_version_id})")
    else:
        print("Ledger version maps:    unknown")

    if (
        training_state_maps is not None
        and ledger_maps is not None
        and training_state_maps != ledger_maps
    ):
        print(
            "WARNING: training_state.json and the latest shadow ledger version "
            f"disagree on training maps ({training_state_maps} vs {ledger_maps})."
        )

    if maps_behind >= STALE_MAP_THRESHOLD:
        print(
            "WARNING: Model is "
            f"{maps_behind} maps out of date versus the cleaned canonical scrape "
            f"(threshold: {STALE_MAP_THRESHOLD})."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the daily update flow: scrape recent matches, predict upcoming matches, and refresh the shadow ledger."
    )
    parser.add_argument("--pages", type=int, default=1, help="HLTV results pages to scrape before predicting.")
    parser.add_argument("--matches", "--count", type=int, dest="count", help="Stop scraping after this many new matches.")
    parser.add_argument("--event-id", type=int, help="HLTV event ID for upcoming-match predictions.")
    parser.add_argument("--output", help="Prediction report output path.")
    parser.add_argument("--html-file", help="Local upcoming-matches HTML file for predictions.")
    parser.add_argument("--iters", type=int, help="Monte Carlo iterations for veto simulation.")
    parser.add_argument("--threshold", type=float, help="Probability truncation threshold.")
    parser.add_argument("--no-open", action="store_true", help="Do not open the prediction report in a browser.")
    parser.add_argument("--run-once", action="store_true", help="Run one update pass and exit instead of looping.")
    parser.add_argument("--interval-hours", type=float, default=DEFAULT_INTERVAL_HOURS, help="Base delay between looped runs.")
    parser.add_argument("--jitter-minutes", type=float, default=DEFAULT_JITTER_MINUTES, help="Random +/- jitter applied to each loop delay.")
    return parser.parse_args()


def run_update(args: argparse.Namespace) -> None:
    scrape_command = [
        sys.executable,
        "-m",
        "ingestion.fetch_hltv_matches",
        "--pages",
        str(args.pages),
    ]
    if args.count is not None:
        scrape_command.extend(["--matches", str(args.count)])

    predict_command = [sys.executable, "-m", "model.automate_predictions"]
    if args.event_id is not None:
        predict_command.extend(["--event-id", str(args.event_id)])
    if args.output:
        predict_command.extend(["--output", args.output])
    if args.html_file:
        predict_command.extend(["--html-file", args.html_file])
    if args.iters is not None:
        predict_command.extend(["--iters", str(args.iters)])
    if args.threshold is not None:
        predict_command.extend(["--threshold", str(args.threshold)])
    if args.no_open:
        predict_command.append("--no-open")

    refresh_command = [sys.executable, "-m", "evaluation.shadow_ledger", "refresh"]

    run_step("Scraping recent matches", scrape_command)
    warn_if_model_is_stale()
    run_step("Running predictions with report", predict_command)
    run_step("Refreshing shadow ledger", refresh_command)


def next_delay_seconds(args: argparse.Namespace) -> float:
    interval_seconds = max(args.interval_hours, 0) * 60 * 60
    jitter_seconds = max(args.jitter_minutes, 0) * 60
    delay = interval_seconds + random.uniform(-jitter_seconds, jitter_seconds)
    return max(delay, 60)


def main() -> None:
    args = parse_args()

    if args.run_once:
        run_update(args)
        return

    print(
        "Starting update loop "
        f"(base interval: {args.interval_hours:g}h, jitter: +/- {args.jitter_minutes:g}m). "
        "Press Ctrl+C to stop."
    )

    while True:
        try:
            run_update(args)
        except subprocess.CalledProcessError as exc:
            print(f"WARNING: Update pass failed with exit code {exc.returncode}. Continuing loop.")
        except Exception as exc:
            print(f"WARNING: Update pass failed: {exc}. Continuing loop.")

        delay = next_delay_seconds(args)
        next_run = datetime.now() + timedelta(seconds=delay)
        print(f"\nNext update scheduled for {next_run:%Y-%m-%d %H:%M:%S} after {delay / 60:.1f} minutes.")

        try:
            time.sleep(delay)
        except KeyboardInterrupt:
            print("\nUpdate loop stopped.")
            return


if __name__ == "__main__":
    main()
