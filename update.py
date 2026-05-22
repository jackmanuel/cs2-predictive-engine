import argparse
import json
import random
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from processing.clean import clean_data


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
STALE_MAP_THRESHOLD = 500
HLTV_MATCHES_FILE = DATA_DIR / "raw" / "hltv_matches.json"
TRAINING_STATE_FILE = DATA_DIR / "training_state.json"
DEFAULT_INTERVAL_HOURS = 2.0
DEFAULT_JITTER_MINUTES = 30.0


def run_step(label: str, command: list[str]) -> None:
    print(f"\n=== {label} ===")
    print(" ".join(command))
    subprocess.run(command, check=True)


def load_training_state() -> dict | None:
    if not TRAINING_STATE_FILE.exists():
        return None

    try:
        with open(TRAINING_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARNING: Could not read training state at {TRAINING_STATE_FILE}: {exc}")
        return None


def parse_state_datetime(value: str | None) -> datetime | None:
    if not value:
        return None

    normalized = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None

    return parsed.replace(tzinfo=None)


def is_played_raw_map(map_data: dict) -> bool:
    score1 = str(map_data.get("team1_score", "")).strip()
    score2 = str(map_data.get("team2_score", "")).strip()
    return bool(score1 and score2 and score1 != "-" and score2 != "-")


def count_raw_maps_since(trained_at: datetime) -> int | None:
    if not HLTV_MATCHES_FILE.exists():
        return None

    try:
        with open(HLTV_MATCHES_FILE, "r", encoding="utf-8") as f:
            matches = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARNING: Could not read canonical match data at {HLTV_MATCHES_FILE}: {exc}")
        return None

    new_maps = 0
    for match in matches:
        match_date = parse_state_datetime(match.get("date"))
        if match_date is None or match_date <= trained_at:
            continue

        new_maps += sum(
            1 for map_data in match.get("hltv_maps", []) if is_played_raw_map(map_data)
        )

    return new_maps


def print_model_freshness() -> None:
    state = load_training_state()
    if state is None:
        print("WARNING: Could not determine model training time; skipping model freshness check.")
        return

    trained_at = parse_state_datetime(state.get("training_date"))
    if trained_at is None:
        trained_at = parse_state_datetime((state.get("date_range") or {}).get("end"))
    if trained_at is None:
        print("WARNING: training_state.json has no parseable training time; skipping model freshness check.")
        return

    new_maps = count_raw_maps_since(trained_at)
    if new_maps is None:
        print("WARNING: Could not count new raw canonical maps; skipping model freshness check.")
        return

    print("\n=== Model freshness ===")
    print(f"Model trained at:        {trained_at:%Y-%m-%d %H:%M:%S}")
    print(f"New raw canonical maps:  {new_maps}")

    if new_maps >= STALE_MAP_THRESHOLD:
        print(
            "WARNING: Model is "
            f"{new_maps} raw canonical maps out of date "
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
    parser.add_argument(
        "--stage",
        choices=["all", "scrape", "predict", "resolve"],
        default="all",
        help="Pipeline stage to run: 'scrape' (completed match scrape & clean), 'predict' (predictions), 'resolve' (ledger refresh), or 'all' (entire pipeline)."
    )
    return parser.parse_args()


def run_update(args: argparse.Namespace) -> None:
    stage = getattr(args, "stage", "all")

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

    if stage in {"all", "scrape"}:
        run_step("Scraping recent matches", scrape_command)
        print_model_freshness()
        print("\n=== Refreshing cleaned map data ===")
        clean_data()

    if stage in {"all", "predict"}:
        run_step("Running predictions with report", predict_command)

    if stage in {"all", "resolve"}:
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
