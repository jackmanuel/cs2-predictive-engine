import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from processing.clean import clean_data
from processing.features import feature_pipeline
from model.train import train_model


PHASES = [
    ("cleaning", "Cleaning raw data", clean_data),
    ("features", "Calculating features", feature_pipeline),
    ("training", "Training model", train_model),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Time the full training pipeline phases and write a JSON timing report."
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Run without the interactive confirmation prompt.",
    )
    return parser.parse_args()


def confirm_full_training_run(skip_prompt):
    if skip_prompt:
        return

    print(
        "This will run the full training pipeline, which can take several minutes "
        "and will refresh local training artefacts."
    )
    response = input("Continue? Type 'yes' to proceed: ").strip().lower()
    if response != "yes":
        print("Cancelled.")
        raise SystemExit(1)


def time_phase(slug, label, func):
    print(f"\n[TIMING] {label}...")
    started = perf_counter()
    func()
    seconds = perf_counter() - started
    print(f"[TIMING] {label} finished in {seconds:.2f}s")
    return {
        "phase": slug,
        "label": label,
        "seconds": round(seconds, 2),
    }


def main():
    args = parse_args()
    confirm_full_training_run(args.yes)

    run_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    phase_results = [time_phase(slug, label, func) for slug, label, func in PHASES]
    total_seconds = sum(phase["seconds"] for phase in phase_results)

    for phase in phase_results:
        phase["percentage"] = round((phase["seconds"] / total_seconds) * 100, 1) if total_seconds else 0

    report = {
        "run_started_at": run_started_at,
        "total_seconds": round(total_seconds, 2),
        "phases": phase_results,
    }

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"training_phase_timings_{timestamp}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nTraining phase timing summary:")
    for phase in phase_results:
        print(f"  - {phase['label']}: {phase['seconds']:.2f}s ({phase['percentage']:.1f}%)")
    print(f"  - Total: {total_seconds:.2f}s")
    print(f"\nTiming report written to {report_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
