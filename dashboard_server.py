import json
import mimetypes
import os
import queue
import random
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_ROOT / "reports"
STATIC_DIR = PROJECT_ROOT / "static"
TEMPLATE_PATH = PROJECT_ROOT / "templates" / "dashboard.html"
TRAINING_STATE_PATH = PROJECT_ROOT / "data" / "training_state.json"
PERFORMANCE_REPORT_PATH = REPORTS_DIR / "dashboard_shadow_ledger_report.html"
MODEL_CHECKPOINT_PATH = PROJECT_ROOT / "data" / "checkpoints" / "best_mvp_model.pt"
SCALER_PATH = PROJECT_ROOT / "data" / "checkpoints" / "scaler.pkl"
FORFEIT_MODEL_PATH = PROJECT_ROOT / "data" / "checkpoints" / "forfeit_model.joblib"
CLEAN_MAPS_PATH = PROJECT_ROOT / "data" / "processed" / "clean_maps.parquet"
SCRAPER_STATE_PATH = PROJECT_ROOT / "data" / "scraper_state.json"
VENV_PYTHON_PATH = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
PROJECT_PYTHON = str(VENV_PYTHON_PATH if VENV_PYTHON_PATH.exists() else Path(sys.executable))


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
SCRAPER_ITEM_RE = re.compile(r"\[(\d+)/(\d+)\]\s+(Scraping match|Predicting|Skipping):\s*(.+)")
SCRAPER_RESOLVED_RE = re.compile(r"Resolved (\d+)/(\d+) past matches")
SCRAPER_NEXT_RUN_RE = re.compile(r"Next update scheduled for (.+?) after")

DEFAULT_SCRAPER_SETTINGS = {
    "interval_hours": 2.0,
    "jitter_minutes": 30.0,
    "pages": 1,
    "count": None,
    "stage": "all",
}

# Conservative phase ranges based on one local timing run. The values are only
# used to keep dashboard progress moving between sparse pipeline log lines.
PHASE_ESTIMATES = {
    "starting": {"label": "Starting pipeline", "start": 0, "end": 2, "seconds": 3},
    "cleaning": {"label": "Cleaning raw data", "start": 2, "end": 4, "seconds": 5},
    "features": {"label": "Calculating features", "start": 4, "end": 8, "seconds": 12},
    "training_setup": {"label": "Preparing training data", "start": 8, "end": 12, "seconds": 8},
    "training": {"label": "Training model", "start": 12, "end": 96, "seconds": 260},
    "finalizing": {"label": "Finalizing artifacts", "start": 96, "end": 99, "seconds": 8},
    "completed": {"label": "Completed", "start": 100, "end": 100, "seconds": 1},
    "failed": {"label": "Failed", "start": 100, "end": 100, "seconds": 1},
    "idle": {"label": "Idle", "start": 0, "end": 0, "seconds": 1},
}

SCRAPER_PHASE_ESTIMATES = {
    "idle": {"label": "Idle", "start": 0, "end": 0, "seconds": 1},
    "starting": {"label": "Starting update pipeline", "start": 0, "end": 3, "seconds": 2},
    "completed_matches": {"label": "Scraping completed matches", "start": 3, "end": 38, "seconds": 90},
    "model_freshness": {"label": "Checking model freshness", "start": 38, "end": 42, "seconds": 4},
    "upcoming_matches": {"label": "Scraping upcoming matches and odds", "start": 42, "end": 82, "seconds": 120},
    "ledger": {"label": "Refreshing shadow ledger", "start": 82, "end": 98, "seconds": 35},
    "waiting": {"label": "Waiting for next loop pass", "start": 100, "end": 100, "seconds": 1},
    "completed": {"label": "Completed", "start": 100, "end": 100, "seconds": 1},
    "stopping": {"label": "Stopping", "start": 100, "end": 100, "seconds": 1},
    "stopped": {"label": "Stopped", "start": 100, "end": 100, "seconds": 1},
    "failed": {"label": "Failed", "start": 100, "end": 100, "seconds": 1},
}


class RetrainJob:
    def __init__(self):
        self.lock = threading.Lock()
        self.process = None
        self.thread = None
        self.status = "idle"
        self.progress = 0
        self.started_at = None
        self.finished_at = None
        self.returncode = None
        self.lines = []
        self.subscribers = []
        self.phase = "idle"
        self.phase_started_at = None
        self.phase_started_monotonic = None

    def snapshot(self):
        with self.lock:
            return {
                "status": self.status,
                "progress": self.estimated_progress_locked(),
                "phase": self.phase,
                "phase_label": PHASE_ESTIMATES.get(self.phase, PHASE_ESTIMATES["idle"])["label"],
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "returncode": self.returncode,
                "tail": self.lines[-80:],
            }

    def estimated_progress_locked(self):
        if self.status != "running":
            return self.progress

        estimate = PHASE_ESTIMATES.get(self.phase, PHASE_ESTIMATES["starting"])
        if not self.phase_started_monotonic:
            return max(self.progress, estimate["start"])

        elapsed = max(0, time.monotonic() - self.phase_started_monotonic)
        phase_fraction = min(elapsed / estimate["seconds"], 0.98)
        phase_progress = estimate["start"] + ((estimate["end"] - estimate["start"]) * phase_fraction)
        return int(max(self.progress, min(estimate["end"], phase_progress)))

    def set_phase_locked(self, phase):
        if phase == self.phase:
            return

        self.phase = phase
        self.phase_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.phase_started_monotonic = time.monotonic()
        estimate = PHASE_ESTIMATES.get(phase, PHASE_ESTIMATES["starting"])
        self.progress = max(self.progress, estimate["start"])

    def subscribe(self):
        subscriber = queue.Queue()
        with self.lock:
            self.subscribers.append(subscriber)
        return subscriber

    def unsubscribe(self, subscriber):
        with self.lock:
            if subscriber in self.subscribers:
                self.subscribers.remove(subscriber)

    def publish(self, event, payload):
        message = {"event": event, "payload": payload}
        with self.lock:
            subscribers = list(self.subscribers)
        for subscriber in subscribers:
            subscriber.put(message)

    def append_line(self, line):
        clean_line = ANSI_RE.sub("", line).strip()
        if not clean_line:
            return

        with self.lock:
            self.lines.append(clean_line)
            self.lines = self.lines[-400:]
            phase = infer_phase(clean_line)
            if phase:
                self.set_phase_locked(phase)
            floor = infer_progress_floor(clean_line)
            if floor is not None:
                self.progress = max(self.progress, floor)
            progress = self.estimated_progress_locked()
            phase_label = PHASE_ESTIMATES.get(self.phase, PHASE_ESTIMATES["idle"])["label"]

        self.publish("line", {"line": clean_line, "progress": progress, "phase": self.phase, "phase_label": phase_label})

    def start(self):
        with self.lock:
            if self.status == "running":
                return False

            self.status = "running"
            self.progress = 2
            self.started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.finished_at = None
            self.returncode = None
            self.lines = []
            self.phase = "starting"
            self.phase_started_at = self.started_at
            self.phase_started_monotonic = time.monotonic()

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            self.process = subprocess.Popen(
                [PROJECT_PYTHON, "-u", "pipeline.py"],
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            self.thread = threading.Thread(target=self._reader, daemon=True)
            self.thread.start()

        self.publish("status", self.snapshot())
        return True

    def _reader(self):
        process = self.process
        try:
            for line in process.stdout:
                self.append_line(line)
            returncode = process.wait()
        except Exception as exc:
            self.append_line(f"Dashboard server error while reading pipeline output: {exc}")
            returncode = 1

        with self.lock:
            self.returncode = returncode
            self.status = "completed" if returncode == 0 else "failed"
            self.progress = 100
            self.phase = "completed" if returncode == 0 else "failed"
            self.finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.publish("status", self.snapshot())


def default_scraper_counts():
    return {
        "completed_matches": {"current": 0, "total": None, "label": "Completed matches"},
        "upcoming_matches": {"current": 0, "total": None, "label": "Upcoming matches"},
        "ledger": {"current": 0, "total": None, "label": "Ledger matches"},
    }


def load_scraper_state():
    state = {}
    if SCRAPER_STATE_PATH.exists():
        try:
            with SCRAPER_STATE_PATH.open("r", encoding="utf-8") as handle:
                state = json.load(handle)
        except (OSError, json.JSONDecodeError):
            state = {}
    settings = DEFAULT_SCRAPER_SETTINGS | (state.get("settings") or {})
    last_times = {
        "completed_matches": None,
        "upcoming_matches": None,
        "ledger": None,
    }
    last_times.update(state.get("last_times") or {})
    return {"settings": settings, "last_times": last_times}


def write_scraper_state(settings, last_times):
    SCRAPER_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCRAPER_STATE_PATH.write_text(
        json.dumps({"settings": settings, "last_times": last_times}, indent=2),
        encoding="utf-8",
    )


def parse_optional_positive_int(value, default=None, minimum=1, maximum=200):
    if value in ("", None):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def parse_non_negative_float(value, default, maximum):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(maximum, parsed))


class ScraperJob:
    def __init__(self):
        persisted = load_scraper_state()
        self.lock = threading.Lock()
        self.process = None
        self.thread = None
        self.status = "idle"
        self.mode = "once"
        self.progress = 0
        self.started_at = None
        self.finished_at = None
        self.returncode = None
        self.lines = []
        self.subscribers = []
        self.phase = "idle"
        self.phase_started_at = None
        self.phase_started_monotonic = None
        self.current_detail = "No scraper run started."
        self.next_run_at = None
        self.settings = persisted["settings"]
        self.last_times = persisted["last_times"]
        self.counts = default_scraper_counts()

    def snapshot(self):
        with self.lock:
            return self.snapshot_locked()

    def snapshot_locked(self):
        return {
            "status": self.status,
            "mode": self.mode,
            "progress": self.estimated_progress_locked(),
            "phase": self.phase,
            "phase_label": SCRAPER_PHASE_ESTIMATES.get(self.phase, SCRAPER_PHASE_ESTIMATES["idle"])["label"],
            "detail": self.current_detail,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "returncode": self.returncode,
            "next_run_at": self.next_run_at,
            "settings": self.settings,
            "last_times": self.last_times,
            "counts": self.counts,
            "tail": self.lines[-120:],
        }

    def estimated_progress_locked(self):
        if self.status != "running":
            return self.progress
        estimate = SCRAPER_PHASE_ESTIMATES.get(self.phase, SCRAPER_PHASE_ESTIMATES["starting"])
        if not self.phase_started_monotonic or estimate["start"] == estimate["end"]:
            return max(self.progress, estimate["start"])
        elapsed = max(0, time.monotonic() - self.phase_started_monotonic)
        phase_fraction = min(elapsed / estimate["seconds"], 0.92)
        phase_progress = estimate["start"] + ((estimate["end"] - estimate["start"]) * phase_fraction)
        return int(max(self.progress, min(estimate["end"], phase_progress)))

    def subscribe(self):
        subscriber = queue.Queue()
        with self.lock:
            self.subscribers.append(subscriber)
        return subscriber

    def unsubscribe(self, subscriber):
        with self.lock:
            if subscriber in self.subscribers:
                self.subscribers.remove(subscriber)

    def publish(self, event, payload):
        message = {"event": event, "payload": payload}
        with self.lock:
            subscribers = list(self.subscribers)
        for subscriber in subscribers:
            subscriber.put(message)

    def coerce_settings(self, request):
        settings = dict(self.settings)
        settings["interval_hours"] = parse_non_negative_float(
            request.get("interval_hours"), DEFAULT_SCRAPER_SETTINGS["interval_hours"], 24
        )
        settings["jitter_minutes"] = parse_non_negative_float(
            request.get("jitter_minutes"), DEFAULT_SCRAPER_SETTINGS["jitter_minutes"], 240
        )
        settings["pages"] = parse_positive_int(
            request.get("pages"), DEFAULT_SCRAPER_SETTINGS["pages"], minimum=1, maximum=20
        )
        settings["count"] = parse_optional_positive_int(request.get("count"), default=None, maximum=200)
        
        stage = request.get("stage")
        if stage in {"all", "scrape", "predict", "resolve"}:
            settings["stage"] = stage
        else:
            settings["stage"] = DEFAULT_SCRAPER_SETTINGS.get("stage", "all")
            
        return settings

    def start(self, mode, settings):
        with self.lock:
            if self.status in {"running", "stopping"}:
                return False

            self.status = "running"
            self.mode = mode
            self.progress = 1
            self.started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.finished_at = None
            self.returncode = None
            self.lines = []
            self.phase = "starting"
            self.phase_started_at = self.started_at
            self.phase_started_monotonic = time.monotonic()
            self.current_detail = "Starting update.py."
            self.next_run_at = None
            self.settings = settings
            self.counts = default_scraper_counts()
            write_scraper_state(self.settings, self.last_times)

            command = [PROJECT_PYTHON, "-u", "update.py", "--pages", str(settings["pages"]), "--no-open"]
            if settings.get("count") is not None:
                command.extend(["--matches", str(settings["count"])])
            
            stage = settings.get("stage", "all")
            command.extend(["--stage", stage])

            if mode == "once":
                command.append("--run-once")
            else:
                command.extend(
                    [
                        "--interval-hours",
                        str(settings["interval_hours"]),
                        "--jitter-minutes",
                        str(settings["jitter_minutes"]),
                    ]
                )

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            self.process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            self.thread = threading.Thread(target=self._reader, daemon=True)
            self.thread.start()

        self.publish("status", self.snapshot())
        return True

    def stop(self):
        with self.lock:
            if self.status != "running" or self.process is None:
                return False
            self.status = "stopping"
            self.phase = "stopping"
            self.current_detail = "Stopping scraper loop."
            process = self.process

        process.terminate()
        self.publish("status", self.snapshot())
        return True

    def set_phase_locked(self, phase, detail=None):
        if phase != self.phase:
            self.mark_phase_completed_locked(self.phase)
            self.phase = phase
            self.phase_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.phase_started_monotonic = time.monotonic()
            estimate = SCRAPER_PHASE_ESTIMATES.get(phase, SCRAPER_PHASE_ESTIMATES["starting"])
            if phase == "completed_matches":
                self.counts = default_scraper_counts()
                self.next_run_at = None
                self.progress = estimate["start"]
            elif phase == "waiting":
                self.progress = 100
            else:
                self.progress = max(self.progress, estimate["start"])
        if detail:
            self.current_detail = detail

    def mark_phase_completed_locked(self, phase):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if phase == "completed_matches":
            self.last_times["completed_matches"] = now
        elif phase == "upcoming_matches":
            self.last_times["upcoming_matches"] = now
        elif phase == "ledger":
            self.last_times["ledger"] = now
        else:
            return
        write_scraper_state(self.settings, self.last_times)

    def update_count_locked(self, key, current=None, total=None):
        item = self.counts[key]
        if total is not None:
            item["total"] = total
        if current is not None:
            item["current"] = current
        estimate = SCRAPER_PHASE_ESTIMATES.get(key)
        if estimate and item["total"]:
            ratio = max(0, min(1, item["current"] / item["total"]))
            self.progress = max(self.progress, int(estimate["start"] + ((estimate["end"] - estimate["start"]) * ratio)))

    def append_line(self, line):
        clean_line = ANSI_RE.sub("", line).strip()
        if not clean_line:
            return

        with self.lock:
            self.lines.append(clean_line)
            self.lines = self.lines[-500:]
            infer_scraper_progress(self, clean_line)
            snapshot = self.snapshot_locked()

        self.publish("line", {"line": clean_line, "snapshot": snapshot})

    def _reader(self):
        process = self.process
        try:
            for line in process.stdout:
                self.append_line(line)
            returncode = process.wait()
        except Exception as exc:
            self.append_line(f"Dashboard server error while reading update output: {exc}")
            returncode = 1

        with self.lock:
            self.returncode = returncode
            self.finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if self.status == "stopping":
                self.status = "stopped"
                self.phase = "stopped"
                self.current_detail = "Scraper loop stopped."
            else:
                self.status = "completed" if returncode == 0 else "failed"
                self.phase = "completed" if returncode == 0 else "failed"
                self.current_detail = "Update pipeline completed." if returncode == 0 else "Update pipeline failed."
                if returncode == 0:
                    stage = self.settings.get("stage", "all")
                    if stage == "scrape":
                        self.mark_phase_completed_locked("completed_matches")
                    elif stage == "predict":
                        self.mark_phase_completed_locked("upcoming_matches")
                    elif stage == "resolve":
                        self.mark_phase_completed_locked("ledger")
                    else: # "all"
                        self.mark_phase_completed_locked("ledger")
            self.progress = 100
            self.process = None
            snapshot = self.snapshot_locked()

        self.publish("status", snapshot)


retrain_job = RetrainJob()
scraper_job = ScraperJob()
playground_lock = threading.Lock()
playground_predictor_cache = {"signature": None, "ctx": None}
playground_forfeit_cache = {"signature": None, "ctx": None}


def infer_phase(line):
    lowered = line.lower()
    if "phase 1" in lowered or "cleaning" in lowered:
        return "cleaning"
    if "phase 2" in lowered or "feature engineering" in lowered:
        return "features"
    if "applying data mirroring" in lowered or "split sizes" in lowered or "scaler saved" in lowered:
        return "training_setup"
    if "starting ensemble training" in lowered or "training seed" in lowered or "epoch" in lowered:
        return "training"
    if "ensemble test averages" in lowered or "best model globally" in lowered:
        return "finalizing"
    if "phase 3" in lowered or "training" in lowered:
        return "training_setup"
    return None


def infer_progress_floor(line):
    lowered = line.lower()
    if "clean_maps.parquet" in lowered:
        return 4
    if "features.parquet" in lowered or "reduced features saved" in lowered:
        return 8
    if "scaler.pkl" in lowered or "scaler saved" in lowered:
        return 12
    if "training seed 2" in lowered:
        return 29
    if "training seed 3" in lowered:
        return 46
    if "training seed 4" in lowered:
        return 62
    if "training seed 5" in lowered:
        return 79
    if "best_mvp_model.pt" in lowered or "best model globally" in lowered:
        return 97
    if "completed successfully" in lowered:
        return 100
    if "pipeline failed" in lowered:
        return 100
    return None


def infer_scraper_progress(job, line):
    lowered = line.lower()

    if "=== scraping recent matches ===" in lowered:
        job.set_phase_locked("completed_matches", "Scanning HLTV result pages for completed matches.")
        return

    if "=== model freshness ===" in lowered:
        job.set_phase_locked("model_freshness", "Checking how far the model is behind the raw match data.")
        return

    if "=== running predictions with report ===" in lowered:
        job.set_phase_locked("upcoming_matches", "Fetching upcoming matches and bookmaker odds.")
        return

    if "=== refreshing shadow ledger ===" in lowered:
        job.set_phase_locked("ledger", "Resolving completed shadow ledger matches.")
        return

    if "loaded " in lowered and "already scraped matches" in lowered:
        job.current_detail = line
        return

    match = re.search(r"Found (\d+) total new matches", line)
    if match:
        total = int(match.group(1))
        job.update_count_locked("completed_matches", current=0, total=total)
        job.current_detail = f"Found {total} completed matches to scrape."
        return

    match = re.search(r"Preparing to scrape details for (\d+) matches", line)
    if match:
        total = int(match.group(1))
        job.update_count_locked("completed_matches", current=0, total=total)
        job.current_detail = f"Found {total} completed matches to scrape."
        return

    if "no new matches found to scrape" in lowered:
        job.update_count_locked("completed_matches", current=0, total=0)
        job.current_detail = "No new completed matches found."
        return

    item = SCRAPER_ITEM_RE.search(line)
    if item:
        current = int(item.group(1))
        total = int(item.group(2))
        action = item.group(3).lower()
        subject = item.group(4)
        if "scraping match" in action:
            job.set_phase_locked("completed_matches")
            job.update_count_locked("completed_matches", current=current, total=total)
            job.current_detail = f"Scraping completed match {current}/{total}: {subject}"
        else:
            job.set_phase_locked("upcoming_matches")
            job.update_count_locked("upcoming_matches", current=current, total=total)
            verb = "Skipping" if action == "skipping" else "Predicting"
            job.current_detail = f"{verb} upcoming match {current}/{total}: {subject}"
        return

    if "batch complete" in lowered:
        total = job.counts["completed_matches"]["total"]
        if total is not None:
            job.update_count_locked("completed_matches", current=total, total=total)
        job.current_detail = "Completed match scrape finished."
        return

    match = re.search(r"Found (\d+) matches\. Starting predictions", line)
    if match:
        total = int(match.group(1))
        job.update_count_locked("upcoming_matches", current=0, total=total)
        job.current_detail = f"Predicting {total} upcoming matches."
        return

    if "no matches found to predict" in lowered:
        job.update_count_locked("upcoming_matches", current=0, total=0)
        job.current_detail = "No upcoming matches found to predict."
        return

    if "automation complete" in lowered:
        total = job.counts["upcoming_matches"]["total"]
        if total is not None:
            job.update_count_locked("upcoming_matches", current=total, total=total)
        job.current_detail = "Upcoming match scrape and predictions finished."
        return

    match = re.search(r"Resolving (\d+) past matches", line)
    if match:
        total = int(match.group(1))
        job.update_count_locked("ledger", current=0, total=total)
        job.current_detail = f"Updating ledger for {total} completed matches."
        return

    if "no matches ready to resolve" in lowered or "no pending shadow bets" in lowered:
        job.update_count_locked("ledger", current=0, total=0)
        job.current_detail = "No ledger matches ready to refresh."
        return

    if "resolved from canonical data" in lowered or "resolved from hltv" in lowered:
        current = job.counts["ledger"]["current"] + 1
        total = job.counts["ledger"]["total"]
        job.update_count_locked("ledger", current=current, total=total)
        if total:
            job.current_detail = f"Updating ledger match {current}/{total}."
        else:
            job.current_detail = f"Updating ledger match {current}."
        return

    match = SCRAPER_RESOLVED_RE.search(line)
    if match:
        current = int(match.group(1))
        total = int(match.group(2))
        job.update_count_locked("ledger", current=current, total=total)
        job.current_detail = f"Ledger refresh finished: {current}/{total} matches resolved."
        return

    match = SCRAPER_NEXT_RUN_RE.search(line)
    if match:
        job.next_run_at = match.group(1).strip()
        job.set_phase_locked("waiting", f"Next loop pass scheduled for {job.next_run_at}.")
        return


def read_json(path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_training_state(state):
    if not isinstance(state, dict):
        return {}

    normalized = dict(state)
    map_counts = normalized.get("map_popularity") or normalized.get("top_maps")
    if isinstance(map_counts, dict):
        from model.veto_sim import MAP_POOL

        active_map_counts = {
            map_name: int(map_counts.get(map_name, 0))
            for map_name in sorted(MAP_POOL, key=lambda name: map_counts.get(name, 0), reverse=True)
        }
        normalized["map_popularity"] = active_map_counts

    return normalized


def model_info_payload():
    state = normalize_training_state(read_json(TRAINING_STATE_PATH))
    try:
        from evaluation.shadow_ledger import model_version_payload

        model_versions = model_version_payload()
    except Exception as exc:
        model_versions = {"versions": [], "evaluations": [], "error": str(exc)}
    checkpoints = {
        "model_checkpoint": file_meta(PROJECT_ROOT / "data" / "checkpoints" / "best_mvp_model.pt"),
        "scaler": file_meta(PROJECT_ROOT / "data" / "checkpoints" / "scaler.pkl"),
        "features": file_meta(PROJECT_ROOT / "data" / "processed" / "features.parquet"),
        "clean_maps": file_meta(PROJECT_ROOT / "data" / "processed" / "clean_maps.parquet"),
    }
    return {
        "training_state": state,
        "training_state_file": file_meta(TRAINING_STATE_PATH),
        "checkpoints": checkpoints,
        "model_versions": model_versions,
    }


def file_meta(path):
    if not path.exists():
        return {"exists": False, "path": str(path.relative_to(PROJECT_ROOT))}
    stat = path.stat()
    return {
        "exists": True,
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
    }


def latest_prediction_report():
    candidates = sorted(
        REPORTS_DIR.glob("predictions_report*.html"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def existing_performance_report():
    return PERFORMANCE_REPORT_PATH if PERFORMANCE_REPORT_PATH.exists() else None


def generate_performance_report():
    REPORTS_DIR.mkdir(exist_ok=True)
    try:
        from evaluation.shadow_ledger import show_report

        show_report(str(PERFORMANCE_REPORT_PATH), open_browser=False)
        return PERFORMANCE_REPORT_PATH
    except Exception as exc:
        fallback = REPORTS_DIR / "dashboard_performance_error.html"
        fallback.write_text(
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<title>Performance Report Unavailable</title>"
            "<style>body{font-family:system-ui;background:#0d1117;color:#eef4f8;padding:28px}"
            "pre{white-space:pre-wrap;background:#151b23;border:1px solid #2d3744;border-radius:8px;padding:14px}</style>"
            "</head><body><h1>Performance report unavailable</h1>"
            "<p>The shadow ledger report could not be generated.</p>"
            f"<pre>{escape_html(str(exc))}</pre></body></html>",
            encoding="utf-8",
        )
        return fallback


def file_signature(*paths):
    signature = []
    for path in paths:
        if path.exists():
            signature.append((str(path), path.stat().st_mtime_ns, path.stat().st_size))
        else:
            signature.append((str(path), None, None))
    return tuple(signature)


def get_playground_predictor_context():
    from model.predict import PredictorContext

    signature = file_signature(CLEAN_MAPS_PATH, MODEL_CHECKPOINT_PATH, SCALER_PATH)
    with playground_lock:
        if playground_predictor_cache["ctx"] is None or playground_predictor_cache["signature"] != signature:
            playground_predictor_cache["ctx"] = PredictorContext()
            playground_predictor_cache["signature"] = signature
        return playground_predictor_cache["ctx"]


def get_playground_forfeit_context():
    from model.forfeit import ForfeitPredictorContext

    signature = file_signature(FORFEIT_MODEL_PATH)
    with playground_lock:
        if playground_forfeit_cache["ctx"] is None or playground_forfeit_cache["signature"] != signature:
            playground_forfeit_cache["ctx"] = ForfeitPredictorContext()
            playground_forfeit_cache["signature"] = signature
        return playground_forfeit_cache["ctx"]


def playground_options_payload():
    from config import DEFAULT_TEAM_RANK, MC_ITERATIONS, MC_THRESHOLD
    from model.veto_sim import MAP_POOL

    teams = []
    maps = list(MAP_POOL)
    if CLEAN_MAPS_PATH.exists():
        import pandas as pd

        df = pd.read_parquet(CLEAN_MAPS_PATH)
        maps = sorted(set(maps).union(str(name) for name in df.get("map_name", []) if str(name) in set(MAP_POOL)))
        team_rows = []
        for side in ("a", "b"):
            id_col = f"team_{side}_id"
            name_col = f"team_{side}_name"
            rank_col = f"team_{side}_world_rank"
            if id_col not in df.columns:
                continue
            columns = ["date", id_col]
            if name_col in df.columns:
                columns.append(name_col)
            if rank_col in df.columns:
                columns.append(rank_col)
            subset = df[columns].copy()
            subset = subset.rename(columns={id_col: "id", name_col: "name", rank_col: "rank"})
            if "name" not in subset.columns:
                subset["name"] = subset["id"]
            if "rank" not in subset.columns:
                subset["rank"] = DEFAULT_TEAM_RANK
            team_rows.append(subset[["date", "id", "name", "rank"]])

        if team_rows:
            team_df = pd.concat(team_rows, ignore_index=True)
            team_df = team_df.dropna(subset=["id"]).sort_values("date")
            latest = team_df.groupby("id", as_index=False).tail(1)
            teams = [
                {
                    "id": str(row["id"]),
                    "name": str(row["name"] or row["id"]),
                    "rank": int(row["rank"]) if pd.notna(row["rank"]) else DEFAULT_TEAM_RANK,
                }
                for _, row in latest.iterrows()
            ]
            teams.sort(key=lambda item: (item["rank"] if item["rank"] != DEFAULT_TEAM_RANK else 9999, item["name"]))

    return {"teams": teams, "maps": maps, "defaults": {"iterations": MC_ITERATIONS, "threshold": MC_THRESHOLD}}


def normalize_map_names(raw_maps):
    from model.veto_sim import MAP_POOL

    if isinstance(raw_maps, str):
        candidates = [part.strip() for part in re.split(r"[\n,]+", raw_maps) if part.strip()]
    elif isinstance(raw_maps, list):
        candidates = [str(part).strip() for part in raw_maps if str(part).strip()]
    else:
        candidates = []

    canonical = {name.lower(): name for name in MAP_POOL}
    maps = []
    unknown = []
    for candidate in candidates:
        key = candidate.lower()
        if key in canonical:
            maps.append(canonical[key])
        else:
            unknown.append(candidate)
    return maps, unknown


def parse_positive_int(value, default, minimum=1, maximum=50000):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def parse_probability(value, default, minimum=0.1, maximum=1.0):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed > 1:
        parsed = parsed / 100
    return max(minimum, min(maximum, parsed))


def parse_team_choice(value, team_a_id, team_b_id):
    from model.predict import normalize_name

    normalized = normalize_name(value or "")
    if normalized in {"", "RANDOM", "AUTO"}:
        return None
    if normalized in {"A", "TEAM_A"} or normalized == team_a_id:
        return "a"
    if normalized in {"B", "TEAM_B"} or normalized == team_b_id:
        return "b"
    return None


def raw_recent_record(history, current_date, days=None):
    if not history:
        return {"matches": 0, "wins": 0, "win_rate": None}
    cutoff = current_date - __import__("pandas").Timedelta(days=days) if days else None
    outcomes = []
    for item in history:
        if cutoff is not None and item[0] < cutoff:
            continue
        if len(item) >= 3:
            outcomes.append(1 if item[1] > item[2] else 0)
        else:
            outcomes.append(int(item[1]))
    wins = int(sum(outcomes))
    matches = len(outcomes)
    return {"matches": matches, "wins": wins, "win_rate": (wins / matches if matches else None)}


def lan_rate(history, current_date, days):
    if not history:
        return None
    import pandas as pd

    cutoff = current_date - pd.Timedelta(days=days)
    recent = [int(value) for date, value in history if date >= cutoff]
    if not recent:
        return None
    return sum(recent) / len(recent)


def pick_comfort(ctx, team_id, map_name, current_date, days=30):
    import pandas as pd

    cutoff = current_date - pd.Timedelta(days=days)
    picks = len([date for date in ctx.team_fpicks.get(team_id, {}).get(map_name, []) if date >= cutoff])
    total = len([date for date in ctx.team_tseries.get(team_id, []) if date >= cutoff])
    return {"picks": picks, "series": total, "rate": picks / total if total else None}


def natural_team_profile(ctx, team_id, team_label, veto_stats, opponent_id=None):
    import math
    import pandas as pd

    from config import DEFAULT_TEAM_RANK, FORM_WINDOW_DAYS, FORM_WINDOW_DAYS_LONG, MAP_WINDOW_DAYS
    from model.veto_sim import MAP_POOL
    from processing.features import get_sos

    now = pd.to_datetime(datetime.now(timezone.utc))
    rank = ctx.latest_ranks.get(team_id, {}).get("world", DEFAULT_TEAM_RANK)
    histories = ctx.gen_histories.get(team_id, [])
    recent_90 = raw_recent_record(histories, now, FORM_WINDOW_DAYS_LONG)
    recent_30 = raw_recent_record(histories, now, FORM_WINDOW_DAYS)
    recent_all = raw_recent_record(histories, now, None)
    avg_opp_rank_90 = math.exp(get_sos(ctx.sos_histories.get(team_id, []), now, FORM_WINDOW_DAYS_LONG))

    maps = []
    for map_name in MAP_POOL:
        record = raw_recent_record(ctx.map_histories.get(team_id, {}).get(map_name, []), now, MAP_WINDOW_DAYS)
        comfort = pick_comfort(ctx, team_id, map_name, now, FORM_WINDOW_DAYS)
        veto_map = veto_stats.get(map_name, {})
        maps.append(
            {
                "map": map_name,
                "wins": record["wins"],
                "matches": record["matches"],
                "win_rate": record["win_rate"],
                "picks": int(veto_map.get("picks", 0)),
                "pick_rate": comfort["rate"],
                "pick_series": comfort["series"],
            }
        )

    h2h = {}
    if opponent_id:
        h2h_key = tuple(sorted([str(team_id), str(opponent_id)]))
        all_h2h = ctx.h2h_history.get(h2h_key, [])
        for days in (30, 90):
            cutoff = now - pd.Timedelta(days=days)
            recent = [winner for date, winner in all_h2h if date >= cutoff]
            h2h[f"{days}d"] = {
                "wins": sum(1 for winner in recent if winner == str(team_id)),
                "maps": len(recent),
            }
        h2h["all"] = {
            "wins": sum(1 for _, winner in all_h2h if winner == str(team_id)),
            "maps": len(all_h2h),
        }

    return {
        "id": team_id,
        "name": team_label,
        "world_rank": None if rank == DEFAULT_TEAM_RANK else int(rank),
        "rank_fallback": rank == DEFAULT_TEAM_RANK,
        "maps_in_database": len(histories),
        "recent": {"90d": recent_90, "30d": recent_30, "all": recent_all},
        "win_streak": int(ctx.latest_streaks.get(team_id, 0)),
        "lan_rate_30d": lan_rate(ctx.lan_histories.get(team_id, []), now, FORM_WINDOW_DAYS),
        "average_opponent_rank_90d": round(avg_opp_rank_90, 1),
        "veto_series_90d": int(veto_stats.get("metadata", {}).get("total_series", 0)),
        "map_stats": maps,
        "head_to_head": h2h,
    }


def veto_steps_for_format(series_format):
    if series_format == "bo1":
        return [
            ("ban", "a", True), ("ban", "a", True),
            ("ban", "b", True), ("ban", "b", True), ("ban", "b", True),
            ("ban", "a", False),
        ]
    if series_format == "bo5":
        return [
            ("ban", "a", True), ("ban", "b", True),
            ("pick", "a"), ("pick", "b"),
            ("pick", "a"), ("pick", "b"),
        ]
    return [
        ("ban", "a", True), ("ban", "b", True),
        ("pick", "a"), ("pick", "b"),
        ("ban", "a", False), ("ban", "b", False),
    ]


def simulate_veto_actions(stats_a, stats_b, team_a_name, team_b_name, series_format, iters, starts_veto):
    from model import veto_sim

    sequence_counts = {}
    map_counts = {map_name: 0 for map_name in veto_sim.MAP_POOL}

    for _ in range(iters):
        starter = starts_veto if starts_veto in {"a", "b"} else ("a" if random.random() < 0.5 else "b")
        if starter == "a":
            side_stats = {"a": stats_a, "b": stats_b}
            side_names = {"a": team_a_name, "b": team_b_name}
        else:
            side_stats = {"a": stats_b, "b": stats_a}
            side_names = {"a": team_b_name, "b": team_a_name}

        pool = veto_sim.MAP_POOL.copy()
        actions = []
        played_maps = []
        for step in veto_steps_for_format(series_format):
            move_type = step[0]
            side = step[1]
            current = side_stats[side]
            opponent = side_stats["b" if side == "a" else "a"]
            if move_type == "ban":
                weights = veto_sim.get_ban_weight(current, opponent, pool, is_first_ban=step[2])
                map_name = random.choices(pool, weights=weights, k=1)[0]
                pool.remove(map_name)
                actions.append({"team": side_names[side], "action": "bans", "map": map_name})
            else:
                weights = veto_sim.get_pick_weight(current, pool)
                map_name = random.choices(pool, weights=weights, k=1)[0]
                pool.remove(map_name)
                played_maps.append(map_name)
                actions.append({"team": side_names[side], "action": "picks", "map": map_name})

        decider = pool[0]
        played_maps.append(decider)
        actions.append({"team": "Decider", "action": "left", "map": decider})
        for map_name in played_maps:
            map_counts[map_name] += 1

        key = " | ".join(f"{action['team']} {action['action']} {action['map']}" for action in actions)
        if key not in sequence_counts:
            sequence_counts[key] = {"count": 0, "actions": actions, "maps": played_maps}
        sequence_counts[key]["count"] += 1

    top_vetoes = sorted(sequence_counts.values(), key=lambda item: item["count"], reverse=True)[:8]
    return {
        "top_vetoes": [
            {
                "probability": item["count"] / iters if iters else 0,
                "count": item["count"],
                "actions": item["actions"],
                "maps": item["maps"],
            }
            for item in top_vetoes
        ],
        "map_appearance": [
            {"map": map_name, "probability": count / iters if iters else 0, "count": count}
            for map_name, count in sorted(map_counts.items(), key=lambda pair: pair[1], reverse=True)
        ],
    }


def top_map_sequences(sequence_counts, iters, limit=8):
    return [
        {"maps": seq.split(","), "probability": count / iters if iters else 0, "count": count}
        for seq, count in sorted(sequence_counts.items(), key=lambda pair: pair[1], reverse=True)[:limit]
    ]


def bracket_match_payload(round_name, label, left_dist, right_dist, match_format, probability_lookup):
    outcomes = {}
    matchup_rows = []
    for left_team, left_probability in left_dist.items():
        if left_probability <= 0:
            continue
        for right_team, right_probability in right_dist.items():
            if right_probability <= 0:
                continue
            matchup_probability = left_probability * right_probability
            left_win_probability = probability_lookup(left_team, right_team, match_format)
            outcomes[left_team] = outcomes.get(left_team, 0) + matchup_probability * left_win_probability
            outcomes[right_team] = outcomes.get(right_team, 0) + matchup_probability * (1 - left_win_probability)
            matchup_rows.append(
                {
                    "team_a": left_team,
                    "team_b": right_team,
                    "probability": matchup_probability,
                    "team_a_win_probability": left_win_probability,
                }
            )

    return {
        "round": round_name,
        "label": label,
        "format": match_format,
        "left": sorted(left_dist),
        "right": sorted(right_dist),
        "outcomes": outcomes,
        "matchups": sorted(matchup_rows, key=lambda item: item["probability"], reverse=True),
    }


class SwissTeam:
    def __init__(self, name, initial_seed):
        self.name = name
        self.initial_seed = initial_seed
        self.wins = 0
        self.losses = 0
        self.buchholz = 0
        self.opponents = []  # history of opponent names (strings)


def find_swiss_pairings(unpaired_teams, history):
    if not unpaired_teams:
        return []
    team_a = unpaired_teams[0]
    for i in range(len(unpaired_teams) - 1, 0, -1):
        team_b = unpaired_teams[i]
        if team_b.name not in history[team_a.name]:
            remaining = [t for t in unpaired_teams if t != team_a and t != team_b]
            sub_pairings = find_swiss_pairings(remaining, history)
            if sub_pairings is not None:
                return [(team_a, team_b)] + sub_pairings
    return None


def optimize_pickem(teams, run_records, iters):
    import itertools
    
    team_to_idx = {name: i for i, name in enumerate(teams)}
    
    # Precompute masks
    team_3_0_mask = [0] * 16
    team_0_3_mask = [0] * 16
    team_qual_mask = [0] * 16
    
    for i, run_record in enumerate(run_records):
        for team_name, record in run_record.items():
            t_idx = team_to_idx[team_name]
            if record == "3-0":
                team_3_0_mask[t_idx] |= (1 << i)
            elif record == "0-3":
                team_0_3_mask[t_idx] |= (1 << i)
            elif record in ("3-1", "3-2"):
                team_qual_mask[t_idx] |= (1 << i)
                
    def add3(a, b, c):
        s = a ^ b ^ c
        cy = (a & b) | (c & (a ^ b))
        return s, cy

    def add2(a, b):
        return a ^ b, a & b

    best_prob = -1.0
    best_picks = None
    
    # Fully exhaustive search over all 10,090,080 combinations without any pruning
    for p_30 in itertools.combinations(range(16), 2):
        x0 = team_3_0_mask[p_30[0]]
        x1 = team_3_0_mask[p_30[1]]
        
        remaining_after_30 = tuple(t for t in range(16) if t not in p_30)
        for p_03 in itertools.combinations(remaining_after_30, 2):
            x2 = team_0_3_mask[p_03[0]]
            x3 = team_0_3_mask[p_03[1]]
            
            remaining_after_03 = tuple(t for t in remaining_after_30 if t not in p_03)
            
            for p_qual in itertools.combinations(remaining_after_03, 6):
                x4 = team_qual_mask[p_qual[0]]
                x5 = team_qual_mask[p_qual[1]]
                x6 = team_qual_mask[p_qual[2]]
                x7 = team_qual_mask[p_qual[3]]
                x8 = team_qual_mask[p_qual[4]]
                x9 = team_qual_mask[p_qual[5]]
                
                s1_a, c2_a = add3(x0, x1, x2)
                s1_b, c2_b = add3(x3, x4, x5)
                s1_c, c2_c = add3(x6, x7, x8)
                
                s1_d, c2_d = add3(s1_a, s1_b, s1_c)
                s0, c2_e = add2(s1_d, x9)
                
                s2_a, c4_a = add3(c2_a, c2_b, c2_c)
                s1, c4_b = add3(s2_a, c2_d, c2_e)
                
                s2, s3 = add2(c4_a, c4_b)
                
                ge_5_mask = s3 | (s2 & (s1 | s0))
                
                successes = ge_5_mask.bit_count()
                prob = successes / iters
                
                if prob > best_prob:
                    best_prob = prob
                    best_picks = {
                        "picks_3_0": [teams[p_30[0]], teams[p_30[1]]],
                        "picks_0_3": [teams[p_03[0]], teams[p_03[1]]],
                        "picks_qual": [teams[p_qual[0]], teams[p_qual[1]], teams[p_qual[2]], teams[p_qual[3]], teams[p_qual[4]], teams[p_qual[5]]]
                    }
                    
    return {
        "success_probability": float(best_prob),
        "picks_3_0": best_picks["picks_3_0"] if best_picks else [],
        "picks_0_3": best_picks["picks_0_3"] if best_picks else [],
        "picks_qual": best_picks["picks_qual"] if best_picks else []
    }


def simulate_swiss_stage(teams, series_format, threshold, iters, probability_lookup, bo3_only=False):
    import random
    
    qualifications = {team: 0 for team in teams}
    record_counts = {team: {
        "3-0": 0, "3-1": 0, "3-2": 0,
        "2-3": 0, "1-3": 0, "0-3": 0
    } for team in teams}
    
    run_records = []
    
    slot_matchups = {}  # (round, record_key, slot_idx) -> { (team_a, team_b): count }
    slot_winners = {}   # (round, record_key, slot_idx) -> { (team_a, team_b): count_t_a_won }
    
    for _ in range(iters):
        team_objs = [SwissTeam(name, idx) for idx, name in enumerate(teams)]
        team_map = {t.name: t for t in team_objs}
        
        for r in range(1, 6):
            active = [t for t in team_objs if t.wins < 3 and t.losses < 3]
            if not active:
                break
                
            # Update Buchholz scores (wins - losses of faced opponents)
            for t in active:
                t.buchholz = sum((team_map[opp_name].wins - team_map[opp_name].losses) for opp_name in t.opponents)
                
            # Group by current record
            pools = {}
            for t in active:
                key = f"{t.wins}-{t.losses}"
                pools.setdefault(key, []).append(t)
                
            for record_key, pool_teams in pools.items():
                # Sort pool: primary Buchholz descending, secondary Initial Seed ascending
                pool_teams.sort(key=lambda t: (-t.buchholz, t.initial_seed))
                
                if r == 1:
                    # Round 1 in Swiss Major stages pairs teams using 1 vs 9, 2 vs 10, etc. (i vs i + 8)
                    pairings = []
                    n = len(pool_teams)
                    for i in range(n // 2):
                        pairings.append((pool_teams[i], pool_teams[i + n // 2]))
                else:
                    # Pair subsequent rounds using the recursive backtracking algorithm
                    pairings = find_swiss_pairings(pool_teams, {t.name: set(t.opponents) for t in pool_teams})
                    if pairings is None:
                        # Fallback: simple folded pairing to prevent deadlock
                        pairings = []
                        n = len(pool_teams)
                        for i in range(n // 2):
                            pairings.append((pool_teams[i], pool_teams[n - 1 - i]))
                        
                for slot_idx, (t_a, t_b) in enumerate(pairings):
                    # Deciders (wins=2 or losses=2) are BO3, others BO1
                    if bo3_only:
                        m_fmt = "bo3"
                    elif series_format == "bo1":
                        m_fmt = "bo1"
                    elif series_format == "bo5":
                        m_fmt = "bo5"
                    else:
                        m_fmt = "bo3" if (t_a.wins == 2 or t_a.losses == 2) else "bo1"
                        
                    prob_a = probability_lookup(t_a.name, t_b.name, m_fmt)
                    
                    if random.random() < prob_a:
                        winner, loser = t_a, t_b
                        is_a_winner = True
                    else:
                        winner, loser = t_b, t_a
                        is_a_winner = False
                        
                    winner.wins += 1
                    loser.losses += 1
                    t_a.opponents.append(t_b.name)
                    t_b.opponents.append(t_a.name)
                    
                    stat_key = (r, record_key, slot_idx)
                    matchup_pair = (t_a.name, t_b.name)
                    
                    slot_matchups.setdefault(stat_key, {})
                    slot_matchups[stat_key][matchup_pair] = slot_matchups[stat_key].get(matchup_pair, 0) + 1
                    
                    slot_winners.setdefault(stat_key, {})
                    if is_a_winner:
                        slot_winners[stat_key][matchup_pair] = slot_winners[stat_key].get(matchup_pair, 0) + 1
                        
        run_record = {}
        for t in team_objs:
            if t.wins == 3:
                qualifications[t.name] += 1
                record_counts[t.name][f"3-{t.losses}"] += 1
                run_record[t.name] = f"3-{t.losses}"
            elif t.losses == 3:
                record_counts[t.name][f"{t.wins}-3"] += 1
                run_record[t.name] = f"{t.wins}-3"
        run_records.append(run_record)
                
    # Build rounds structure
    rounds = []
    for r in range(1, 6):
        round_matches = []
        if r == 1:
            pool_keys = ["0-0"]
        elif r == 2:
            pool_keys = ["1-0", "0-1"]
        elif r == 3:
            pool_keys = ["2-0", "1-1", "0-2"]
        elif r == 4:
            pool_keys = ["2-1", "1-2"]
        elif r == 5:
            pool_keys = ["2-2"]
            
        for record_key in pool_keys:
            if record_key == "0-0":
                num_slots = 8
            elif record_key in {"1-0", "0-1", "1-1"}:
                num_slots = 4
            elif record_key in {"2-1", "1-2", "2-2"}:
                num_slots = 3
            elif record_key in {"2-0", "0-2"}:
                num_slots = 2
                
            for slot_idx in range(num_slots):
                stat_key = (r, record_key, slot_idx)
                
                if bo3_only:
                    m_fmt = "bo3"
                elif series_format == "bo1":
                    m_fmt = "bo1"
                elif series_format == "bo5":
                    m_fmt = "bo5"
                else:
                    m_fmt = "bo3" if record_key in {"2-0", "0-2", "2-1", "1-2", "2-2"} else "bo1"
                    
                matchup_rows = []
                outcomes = {}
                left_teams = set()
                right_teams = set()
                
                for (t_a, t_b), count in slot_matchups.get(stat_key, {}).items():
                    matchup_prob = count / iters
                    win_prob = probability_lookup(t_a, t_b, m_fmt)
                    matchup_rows.append({
                        "team_a": t_a,
                        "team_b": t_b,
                        "probability": matchup_prob,
                        "team_a_win_probability": win_prob
                    })
                    
                    if r == 1:
                        t_a_wins = slot_winners.get(stat_key, {}).get((t_a, t_b), 0)
                        outcomes[t_a] = t_a_wins / iters
                        outcomes[t_b] = (iters - t_a_wins) / iters
                    else:
                        outcomes[t_a] = outcomes.get(t_a, 0.0) + matchup_prob
                        outcomes[t_b] = outcomes.get(t_b, 0.0) + matchup_prob
                        
                    left_teams.add(t_a)
                    right_teams.add(t_b)
                    
                matchup_rows.sort(key=lambda m: m["probability"], reverse=True)
                
                label = f"{record_key} Match {slot_idx + 1}"
                if r == 1:
                    label = f"Match {slot_idx + 1}"
                    
                round_matches.append({
                    "round": f"round_{r}",
                    "pool": record_key,
                    "label": label,
                    "format": m_fmt.upper(),
                    "left": sorted(left_teams),
                    "right": sorted(right_teams),
                    "outcomes": outcomes,
                    "matchups": matchup_rows
                })
                
        rounds.append({
            "name": f"Round {r}",
            "matches": round_matches
        })
        
    champion_probabilities = [
        {"team": team, "probability": qualifications[team] / iters, "records": record_counts[team]}
        for team in sorted(qualifications, key=lambda t: qualifications[t], reverse=True)
    ]
    
    pickem_optimization = optimize_pickem(teams, run_records, iters)
    
    return rounds, champion_probabilities, pickem_optimization


def bracket_simulation_payload(request):
    from config import MC_ITERATIONS, MC_THRESHOLD
    from model.predict import calculate_expected_series_win

    raw_teams = request.get("teams", [])
    if isinstance(raw_teams, str):
        teams = [part.strip() for part in re.split(r"[\n,]+", raw_teams) if part.strip()]
    else:
        teams = [str(team).strip() for team in raw_teams if str(team).strip()]

    bracket_format = str(request.get("format", "8")).lower().replace("team", "").strip()
    if bracket_format in {"16", "16-team", "swiss", "16_team"}:
        team_count = 16
    elif bracket_format in {"6", "6-team", "6_team"}:
        team_count = 6
    else:
        team_count = 8

    if len(teams) != team_count:
        if team_count == 16:
            return {"error": "Enter exactly 16 teams for the Swiss stage simulation."}, HTTPStatus.BAD_REQUEST
        return {"error": f"Enter exactly {team_count} teams for a {team_count}-team playoff."}, HTTPStatus.BAD_REQUEST

    if len({team.casefold() for team in teams}) != len(teams):
        return {"error": "Each slot needs a unique team."}, HTTPStatus.BAD_REQUEST

    series_format = str(request.get("series_format", "bo3")).lower()
    if series_format not in {"bo1", "bo3", "bo5"}:
        series_format = "bo3"
    grand_final_format = str(request.get("grand_final_format", "bo5")).lower()
    if grand_final_format not in {"bo1", "bo3", "bo5"}:
        grand_final_format = "bo5"
    iters = parse_positive_int(request.get("iters"), MC_ITERATIONS, minimum=100, maximum=25000)
    threshold = parse_probability(request.get("threshold"), MC_THRESHOLD)
    ctx = get_playground_predictor_context()
    matchup_cache = {}

    def probability_lookup(team_a, team_b, match_format):
        key = (team_a, team_b, match_format)
        if key not in matchup_cache:
            result = calculate_expected_series_win(
                team_a,
                team_b,
                series_format=match_format,
                threshold=threshold,
                iters=iters,
                starts_veto=None,
                ctx=ctx,
            )
            matchup_cache[key] = float(result["expected_win_prob"])
        return matchup_cache[key]

    bo3_only = bool(request.get("bo3_only", False))
    pickem_optimization = None
    if team_count == 16:
        rounds, champion_probabilities, pickem_optimization = simulate_swiss_stage(teams, series_format, threshold, iters, probability_lookup, bo3_only=bo3_only)
    elif team_count == 6:
        quarter_1 = bracket_match_payload("quarterfinal", "Quarter-final 1", {teams[2]: 1.0}, {teams[5]: 1.0}, series_format, probability_lookup)
        quarter_2 = bracket_match_payload("quarterfinal", "Quarter-final 2", {teams[3]: 1.0}, {teams[4]: 1.0}, series_format, probability_lookup)
        semi_1 = bracket_match_payload("semifinal", "Semi-final 1", {teams[0]: 1.0}, quarter_1["outcomes"], series_format, probability_lookup)
        semi_2 = bracket_match_payload("semifinal", "Semi-final 2", {teams[1]: 1.0}, quarter_2["outcomes"], series_format, probability_lookup)
        final = bracket_match_payload("final", "Grand final", semi_1["outcomes"], semi_2["outcomes"], grand_final_format, probability_lookup)
        rounds = [
            {"name": "Quarter-finals", "matches": [quarter_1, quarter_2]},
            {"name": "Semi-finals", "matches": [semi_1, semi_2]},
            {"name": "Grand final", "matches": [final]},
        ]
        champion_probabilities = [
            {"team": team, "probability": probability}
            for team, probability in sorted(final["outcomes"].items(), key=lambda item: item[1], reverse=True)
        ]
    else:
        quarters = [
            bracket_match_payload("quarterfinal", f"Quarter-final {index + 1}", {teams[index * 2]: 1.0}, {teams[index * 2 + 1]: 1.0}, series_format, probability_lookup)
            for index in range(4)
        ]
        semi_1 = bracket_match_payload("semifinal", "Semi-final 1", quarters[0]["outcomes"], quarters[1]["outcomes"], series_format, probability_lookup)
        semi_2 = bracket_match_payload("semifinal", "Semi-final 2", quarters[2]["outcomes"], quarters[3]["outcomes"], series_format, probability_lookup)
        final = bracket_match_payload("final", "Grand final", semi_1["outcomes"], semi_2["outcomes"], grand_final_format, probability_lookup)
        rounds = [
            {"name": "Quarter-finals", "matches": quarters},
            {"name": "Semi-finals", "matches": [semi_1, semi_2]},
            {"name": "Grand final", "matches": [final]},
        ]
        champion_probabilities = [
            {"team": team, "probability": probability}
            for team, probability in sorted(final["outcomes"].items(), key=lambda item: item[1], reverse=True)
        ]

    return {
        "format": team_count,
        "settings": {
            "series_format": series_format,
            "grand_final_format": grand_final_format,
            "iterations": iters,
            "threshold": threshold,
            "bo3_only": bo3_only,
        },
        "teams": teams,
        "rounds": rounds,
        "champion_probabilities": champion_probabilities,
        "pickem_optimization": pickem_optimization,
        "matchup_count": len(matchup_cache),
    }, HTTPStatus.OK


def playground_prediction_payload(request):
    from config import MC_ITERATIONS, MC_THRESHOLD
    from model import veto_sim
    from model.forfeit import polymarket_fair_probs, predict_forfeit_probability
    from model.predict import (
        calculate_expected_series_win,
        combine_probs,
        get_win_probabilities,
        normalize_name,
        scoreline_probabilities,
    )

    team_a_raw = str(request.get("team_a", "")).strip()
    team_b_raw = str(request.get("team_b", "")).strip()
    if not team_a_raw or not team_b_raw:
        return {"error": "Enter both teams."}, HTTPStatus.BAD_REQUEST

    series_format = str(request.get("format", "bo3")).lower()
    if series_format not in {"bo1", "bo3", "bo5"}:
        series_format = "bo3"
    bo = int(series_format.replace("bo", ""))
    iters = parse_positive_int(request.get("iters"), MC_ITERATIONS, minimum=100, maximum=25000)
    threshold = parse_probability(request.get("threshold"), MC_THRESHOLD)

    team_a_id = normalize_name(team_a_raw)
    team_b_id = normalize_name(team_b_raw)
    pick_first = parse_team_choice(request.get("pick_first"), team_a_id, team_b_id)

    maps, unknown_maps = normalize_map_names(request.get("maps", ""))
    if series_format == "bo1":
        maps = []
    if unknown_maps:
        return {"error": f"Unknown map name: {', '.join(unknown_maps)}."}, HTTPStatus.BAD_REQUEST
    if maps and len(maps) < bo:
        return {"error": f"Not enough maps provided. {series_format.upper()} requires {bo} maps."}, HTTPStatus.BAD_REQUEST
    if len(maps) > bo:
        return {"error": f"{series_format.upper()} accepts at most {bo} maps."}, HTTPStatus.BAD_REQUEST

    single_map = str(request.get("single_map", "")).strip()
    single_maps, unknown_single = normalize_map_names(single_map)
    if unknown_single:
        return {"error": f"Unknown map name: {', '.join(unknown_single)}."}, HTTPStatus.BAD_REQUEST
    single_map = single_maps[0] if single_maps and series_format == "bo1" else None
    single_picker = parse_team_choice(request.get("single_picker"), team_a_id, team_b_id)

    ctx = get_playground_predictor_context()
    veto_df = veto_sim.load_data()
    stats_a = veto_sim.get_team_stats(team_a_id, veto_df)
    stats_b = veto_sim.get_team_stats(team_b_id, veto_df)

    veto_payload = simulate_veto_actions(
        stats_a, stats_b, team_a_raw, team_b_raw, series_format, iters, pick_first
    )

    map_prediction = None
    if single_map:
        picker_override = "neutral"
        if single_picker == "a":
            picker_override = "team_a"
        elif single_picker == "b":
            picker_override = "team_b"
        prob = get_win_probabilities(ctx, team_a_id, team_b_id, [single_map], picker_override=picker_override)[0]
        map_prediction = {
            "map": single_map,
            "team_a_probability": prob,
            "team_b_probability": 1 - prob,
            "picker": "team_a" if single_picker == "a" else "team_b" if single_picker == "b" else "neutral",
        }

    series_map_details = []
    if series_format == "bo1" and single_map and map_prediction:
        series_prob = map_prediction["team_a_probability"]
        score_probabilities = scoreline_probabilities([series_prob], bo)
        starter_label = "50/50" if pick_first is None else team_a_raw if pick_first == "a" else team_b_raw
        sequence_source = "provided"
        map_sequences = [{"maps": [single_map], "probability": 1.0, "count": None}]
        series_map_details = [
            {
                "map": single_map,
                "team_a_probability": map_prediction["team_a_probability"],
                "team_b_probability": map_prediction["team_b_probability"],
                "picker": map_prediction["picker"],
            }
        ]
    elif maps:
        if pick_first in {"a", "b"}:
            map_probs = get_win_probabilities(ctx, team_a_id, team_b_id, maps, veto_starter=pick_first)
            series_prob = combine_probs(map_probs, bo)
            score_probabilities = scoreline_probabilities(map_probs, bo)
            starter_label = team_a_raw if pick_first == "a" else team_b_raw
        else:
            map_probs_a = get_win_probabilities(ctx, team_a_id, team_b_id, maps, veto_starter="a")
            map_probs_b = get_win_probabilities(ctx, team_a_id, team_b_id, maps, veto_starter="b")
            series_prob = (combine_probs(map_probs_a, bo) + combine_probs(map_probs_b, bo)) / 2
            score_probs_a = scoreline_probabilities(map_probs_a, bo)
            score_probs_b = scoreline_probabilities(map_probs_b, bo)
            score_probabilities = {
                score: (score_probs_a.get(score, 0) + score_probs_b.get(score, 0)) / 2
                for score in score_probs_a.keys()
            }
            map_probs = [(a + b) / 2 for a, b in zip(map_probs_a, map_probs_b)]
            starter_label = "50/50"
        sequence_source = "provided"
        map_sequences = [{"maps": maps, "probability": 1.0, "count": None}]
        for index, (map_name, prob) in enumerate(zip(maps, map_probs)):
            picker = "neutral"
            if pick_first in {"a", "b"}:
                if bo == 3:
                    if index == 0:
                        picker = "team_a" if pick_first == "a" else "team_b"
                    elif index == 1:
                        picker = "team_b" if pick_first == "a" else "team_a"
                elif bo == 5:
                    if index in {0, 2}:
                        picker = "team_a" if pick_first == "a" else "team_b"
                    elif index in {1, 3}:
                        picker = "team_b" if pick_first == "a" else "team_a"
            series_map_details.append(
                {
                    "map": map_name,
                    "team_a_probability": prob,
                    "team_b_probability": 1 - prob,
                    "picker": picker,
                }
            )
    else:
        results = calculate_expected_series_win(
            team_a_raw,
            team_b_raw,
            series_format=series_format,
            threshold=threshold,
            iters=iters,
            starts_veto=pick_first,
            ctx=ctx,
        )
        series_prob = results["expected_win_prob"]
        score_probabilities = results.get("score_probabilities", {})
        map_probs = []
        starter_label = team_a_raw if pick_first == "a" else team_b_raw if pick_first == "b" else "50/50"
        sequence_source = "simulated"
        map_sequences = top_map_sequences(results["sequence_counts"], iters)

    scorelines = []
    for score, probability in score_probabilities.items():
        team_a_maps, team_b_maps = [int(part) for part in score.split("-", 1)]
        scorelines.append(
            {
                "score": score,
                "probability": probability,
                "winner": "team_a" if team_a_maps > team_b_maps else "team_b",
            }
        )

    forfeit = {"available": False, "error": None}
    try:
        forfeit_ctx = get_playground_forfeit_context()
        rank_a = ctx.latest_ranks.get(team_a_id, {}).get("world")
        rank_b = ctx.latest_ranks.get(team_b_id, {}).get("world")
        settlement_match = {
            "team1": team_a_raw,
            "team2": team_b_raw,
            "event": str(request.get("event", "Manual prediction") or "Manual prediction"),
            "format": series_format,
            "is_lan": bool(request.get("is_lan", False)),
            "team1_rank": rank_a,
            "team2_rank": rank_b,
        }
        forfeit_prob = predict_forfeit_probability(settlement_match, forfeit_ctx)
        fair_a, fair_b = polymarket_fair_probs(series_prob, forfeit_prob)
        forfeit = {
            "available": True,
            "probability": forfeit_prob,
            "fair_team_a": fair_a,
            "fair_team_b": fair_b,
        }
    except Exception as exc:
        forfeit["error"] = str(exc)

    return {
        "teams": {"team_a": team_a_raw, "team_b": team_b_raw, "team_a_id": team_a_id, "team_b_id": team_b_id},
        "settings": {
            "format": series_format,
            "iterations": iters,
            "threshold": threshold,
            "pick_first": starter_label,
            "sequence_source": sequence_source,
        },
        "map_prediction": map_prediction,
        "series_prediction": {
            "team_a_probability": series_prob,
            "team_b_probability": 1 - series_prob,
            "maps": series_map_details,
            "map_sequences": map_sequences,
            "scorelines": scorelines,
        },
        "veto": veto_payload,
        "forfeit": forfeit,
        "team_stats": {
            "team_a": natural_team_profile(ctx, team_a_id, team_a_raw, stats_a, opponent_id=team_b_id),
            "team_b": natural_team_profile(ctx, team_b_id, team_b_raw, stats_b, opponent_id=team_a_id),
        },
    }, HTTPStatus.OK


def escape_html(value):
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "CS2Dashboard/1.0"

    def handle(self):
        try:
            super().handle()
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            pass

    def log_message(self, fmt, *args):
        print(f"{self.address_string()} - {fmt % args}")

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/":
            self.serve_dashboard()
        elif path == "/api/model-info":
            self.send_json(model_info_payload())
        elif path == "/api/playground/options":
            try:
                self.send_json(playground_options_payload())
            except Exception as exc:
                self.send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
        elif path == "/api/retrain/status":
            self.send_json(retrain_job.snapshot())
        elif path == "/api/retrain/events":
            self.serve_retrain_events()
        elif path == "/api/scraper/status":
            self.send_json(scraper_job.snapshot())
        elif path == "/api/scraper/events":
            self.serve_scraper_events()
        elif path == "/reports/latest-predictions":
            report = latest_prediction_report()
            if report:
                self.serve_prediction_report(report)
            else:
                self.send_html_message("No prediction report found", "Run model.automate_predictions to create one.")
        elif path == "/reports/performance":
            report = existing_performance_report()
            if report:
                self.serve_file(report)
            else:
                self.send_html_message(
                    "No performance report found",
                    "Click Refresh on the Performance tab to generate one.",
                )
        elif path.startswith("/static/"):
            self.serve_static(path)
        elif path.startswith("/reports/"):
            self.serve_reports_file(path)
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/retrain":
            started = retrain_job.start()
            status = HTTPStatus.ACCEPTED if started else HTTPStatus.CONFLICT
            self.send_json(retrain_job.snapshot(), status=status)
        elif path in {"/api/scraper/run-once", "/api/scraper/start-loop"}:
            payload = self.read_request_json()
            settings = scraper_job.coerce_settings(payload)
            mode = "loop" if path.endswith("start-loop") else "once"
            started = scraper_job.start(mode, settings)
            status = HTTPStatus.ACCEPTED if started else HTTPStatus.CONFLICT
            self.send_json(scraper_job.snapshot(), status=status)
        elif path == "/api/scraper/stop":
            stopped = scraper_job.stop()
            status = HTTPStatus.ACCEPTED if stopped else HTTPStatus.CONFLICT
            self.send_json(scraper_job.snapshot(), status=status)
        elif path == "/api/performance-refresh":
            report = generate_performance_report()
            self.send_json({"report_url": f"/reports/{report.name}"})
        elif path == "/api/playground/predict":
            try:
                payload = self.read_request_json()
                result, status = playground_prediction_payload(payload)
                self.send_json(result, status=status)
            except Exception as exc:
                self.send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
        elif path == "/api/bracket/simulate":
            try:
                payload = self.read_request_json()
                result, status = bracket_simulation_payload(payload)
                self.send_json(result, status=status)
            except Exception as exc:
                self.send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def serve_dashboard(self):
        html = TEMPLATE_PATH.read_text(encoding="utf-8")
        prediction = latest_prediction_report()
        prediction_label = prediction.name if prediction else "No prediction report found"
        html = html.replace("__PREDICTION_REPORT_LABEL__", escape_html(prediction_label))
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def serve_static(self, request_path):
        relative = request_path.removeprefix("/static/").replace("/", os.sep)
        self.serve_safe_path(STATIC_DIR, relative)

    def serve_reports_file(self, request_path):
        relative = request_path.removeprefix("/reports/").replace("/", os.sep)
        self.serve_safe_path(REPORTS_DIR, relative)

    def serve_prediction_report(self, path):
        html = path.read_text(encoding="utf-8")
        replacements = {
            "--bg-color: #0f172a;": "--bg-color: #0d1117;",
            "--card-bg: #1e293b;": "--card-bg: #151b23;",
            "--text-dim: #94a3b8;": "--text-dim: #94a3ad;",
            "--accent-primary: #38bdf8;": "--accent-primary: #65d1b4;",
            "--accent-secondary: #818cf8;": "--accent-secondary: #f4c95d;",
            "--gold: #fbbf24;": "--gold: #f4c95d;",
            "--toggle-bg: #334155;": "--toggle-bg: #1b2430;",
            "background: #0f172a;": "background: #111820;",
            "background: #0f172a; ": "background: #111820; ",
            "<title>CS2 Series Predictions</title>": "<title>Upcoming Match Report</title>",
            "<h1>CS2 Predictor Pro</h1>": "<h1>Upcoming Match Report</h1>",
            """h1 { 
            font-weight: 800; 
            font-size: 2rem; 
            background: linear-gradient(to right, var(--accent-primary), var(--accent-secondary)); 
            -webkit-background-clip: text; 
            background-clip: text;
            -webkit-text-fill-color: transparent; 
        }""": """h1 { 
            font-weight: 800; 
            font-size: 2rem; 
            color: var(--text-main);
        }""",
        }
        for old, new in replacements.items():
            html = html.replace(old, new)

        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def serve_safe_path(self, root, relative):
        target = (root / relative).resolve()
        try:
            target.relative_to(root.resolve())
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
            return
        if not target.exists() or not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        self.serve_file(target)

    def serve_file(self, path):
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with path.open("rb") as handle:
            self.wfile.write(handle.read())

    def send_html_message(self, title, body):
        html = (
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<style>body{font-family:system-ui;background:#0d1117;color:#eef4f8;padding:28px}</style>"
            f"<title>{escape_html(title)}</title></head><body><h1>{escape_html(title)}</h1>"
            f"<p>{escape_html(body)}</p></body></html>"
        )
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def read_request_json(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def send_json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def serve_retrain_events(self):
        subscriber = retrain_job.subscribe()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        self.write_event("status", retrain_job.snapshot())

        try:
            while True:
                try:
                    message = subscriber.get(timeout=2)
                    self.write_event(message["event"], message["payload"])
                except queue.Empty:
                    self.write_event("heartbeat", retrain_job.snapshot())
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            retrain_job.unsubscribe(subscriber)

    def serve_scraper_events(self):
        subscriber = scraper_job.subscribe()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        self.write_event("status", scraper_job.snapshot())

        try:
            while True:
                try:
                    message = subscriber.get(timeout=2)
                    self.write_event(message["event"], message["payload"])
                except queue.Empty:
                    self.write_event("heartbeat", scraper_job.snapshot())
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            scraper_job.unsubscribe(subscriber)

    def write_event(self, event, payload):
        data = json.dumps(payload, ensure_ascii=False)
        self.wfile.write(f"event: {event}\ndata: {data}\n\n".encode("utf-8"))
        self.wfile.flush()


def run(host="127.0.0.1", port=8765):
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    print(f"CS2 dashboard running at http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


def main():
    port = 8765
    host = "127.0.0.1"
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    run(host=host, port=port)


if __name__ == "__main__":
    main()
