import json
import mimetypes
import os
import queue
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
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


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

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
                [sys.executable, "-u", "pipeline.py"],
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


retrain_job = RetrainJob()


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


def read_json(path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def model_info_payload():
    state = read_json(TRAINING_STATE_PATH) or {}
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


def ensure_performance_report():
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

    def log_message(self, fmt, *args):
        print(f"{self.address_string()} - {fmt % args}")

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/":
            self.serve_dashboard()
        elif path == "/api/model-info":
            self.send_json(model_info_payload())
        elif path == "/api/retrain/status":
            self.send_json(retrain_job.snapshot())
        elif path == "/api/retrain/events":
            self.serve_retrain_events()
        elif path == "/reports/latest-predictions":
            report = latest_prediction_report()
            if report:
                self.serve_prediction_report(report)
            else:
                self.send_html_message("No prediction report found", "Run model.automate_predictions to create one.")
        elif path == "/reports/performance":
            self.serve_file(ensure_performance_report())
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
        elif path == "/api/performance-refresh":
            self.serve_file(ensure_performance_report())
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
