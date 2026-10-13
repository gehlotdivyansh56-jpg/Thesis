#!/usr/bin/env python3
"""
server.py -- local backend for feddrl_dashboard.html
========================================================
Replaces run_dashboard.py when you want dashboard runs persisted to disk
(not just kept in the browser's localStorage). Serves the dashboard HTML
and a tiny JSON API it calls automatically:

    GET  /api/runs      -> list of all stored runs (most recent first)
    POST /api/runs      -> save one run {id, name, ts, cfg, hist}

Every run is appended to results_backend/runs.json (a simple JSON array --
the single source of truth the dashboard reads back), AND written out as
its own folder in the same shape results_io.py uses on the Python side:

    results_backend/<run_name>_<timestamp>/
        config.json
        metrics.csv
        reward.png                 (only if matplotlib is installed)
        handoff_probability.png
        pingpong_probability.png

so you can drop a dashboard run straight next to the real train.py /
federated_train.py results/ folders for your panel.

Usage:
    python server.py
    python server.py --file feddrl_dashboard.html --port 8000

Stdlib only -- no pip install needed. If matplotlib happens to be
installed, per-run PNGs are generated too; if not, you still get
config.json + metrics.csv (and can always use the dashboard's own
"Download Results Folder (.zip)" button for PNGs).
"""

import argparse
import csv
import json
import os
import socket
import socketserver
import threading
import webbrowser
from datetime import datetime
from http.server import SimpleHTTPRequestHandler

LOCK = threading.Lock()


def find_free_port(preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]


def _latest_run_by_algo(runs, algo, scenario):
    candidates = [r for r in runs if r["cfg"]["algorithm"] == algo and r["cfg"]["scenario"] == scenario]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r.get("id", 0))


def write_combined_folder(all_runs: list, scenario: str, backend_dir: str):
    """
    ONE stable folder per scenario -- results_backend/comparison_<scenario>/ --
    holding the LATEST federated run and LATEST baseline run for that
    scenario together: one reward.png, one handoff_probability.png, one
    pingpong_probability.png, each with BOTH curves on it (federated =
    solid line, baseline = dashed line), updated in place every time either
    side gets a new run. No per-run folders, no folder pile-up.
    """
    fed = _latest_run_by_algo(all_runs, "federated", scenario)
    base = _latest_run_by_algo(all_runs, "baseline", scenario)
    present = [r for r in (fed, base) if r is not None]
    if not present:
        return None

    run_dir = os.path.join(backend_dir, f"comparison_{scenario}")
    os.makedirs(run_dir, exist_ok=True)

    config = {r["cfg"]["algorithm"]: r["cfg"] for r in present}
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    max_len = max(len(r["hist"]["round"]) for r in present)
    header = ["communication_round"]
    for r in present:
        algo = r["cfg"]["algorithm"]
        header += [f"reward__{algo}", f"handoff_probability__{algo}", f"pingpong_probability__{algo}"]
    rows = [header]
    for i in range(max_len):
        row = [i + 1]
        for r in present:
            h = r["hist"]
            row += [
                h["reward"][i] if i < len(h["reward"]) else "",
                h["ho_prob"][i] if i < len(h["ho_prob"]) else "",
                h["pingpong_prob"][i] if i < len(h["pingpong_prob"]) else "",
            ]
        rows.append(row)
    with open(os.path.join(run_dir, "metrics.csv"), "w", newline="") as f:
        csv.writer(f).writerows(rows)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def plot(file_name, hist_key, ylabel):
            fig, ax = plt.subplots(figsize=(7.5, 4.8))
            for r in present:
                algo = r["cfg"]["algorithm"]
                is_fed = algo == "federated"
                ax.plot(
                    r["hist"]["round"], r["hist"][hist_key],
                    marker="o", markersize=3, linewidth=2,
                    linestyle="-" if is_fed else "--",
                    color="#4361ee" if is_fed else "#e76f51",
                    label="Federated (proposed)" if is_fed else "Baseline DDQN",
                )
            ax.set_xlabel("Communication Round")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel} -- {scenario}")
            ax.grid(True, linestyle="--", alpha=0.4)
            if len(present) > 1:
                ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(os.path.join(run_dir, file_name), dpi=150)
            plt.close(fig)

        plot("reward.png", "reward", "Reward")
        plot("handoff_probability.png", "ho_prob", "Handoff Probability")
        plot("pingpong_probability.png", "pingpong_prob", "Ping-pong Probability")
    except ImportError:
        pass  # matplotlib not installed -- config.json/metrics.csv are still written

    with open(os.path.join(run_dir, "README.txt"), "w") as f:
        f.write(
            f"Scenario: {scenario}\n"
            f"Updated: {datetime.now()}\n\n"
            "This folder always holds the LATEST federated run and LATEST baseline\n"
            "run for this scenario, combined on one chart per metric (federated =\n"
            "solid line, baseline = dashed line). It is overwritten in place every\n"
            "time a new run for this scenario is saved from the dashboard -- it is\n"
            "not a per-run archive; use the dashboard's own 'Download Results\n"
            "Folder' button on a run if you need an individual run's own folder.\n"
        )

    return run_dir


def make_handler(serve_dir: str, backend_dir: str):
    runs_index_path = os.path.join(backend_dir, "runs.json")

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=serve_dir, **kw)

        def log_message(self, fmt, *a):
            pass  # keep the terminal clean

        def _send_json(self, obj, status=200):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self):
            if self.path == "/api/runs":
                with LOCK:
                    runs = []
                    if os.path.exists(runs_index_path):
                        with open(runs_index_path) as f:
                            runs = json.load(f)
                runs.sort(key=lambda r: r.get("id", 0), reverse=True)
                self._send_json(runs)
                return
            super().do_GET()

        def do_POST(self):
            if self.path == "/api/runs":
                length = int(self.headers.get("Content-Length", 0))
                try:
                    run = json.loads(self.rfile.read(length))
                except Exception as e:
                    self._send_json({"error": str(e)}, status=400)
                    return
                with LOCK:
                    runs = []
                    if os.path.exists(runs_index_path):
                        with open(runs_index_path) as f:
                            runs = json.load(f)
                    runs = [r for r in runs if r.get("id") != run.get("id")]  # de-dupe on re-save
                    runs.append(run)
                    os.makedirs(backend_dir, exist_ok=True)
                    with open(runs_index_path, "w") as f:
                        json.dump(runs, f, indent=2)
                    run_dir = write_combined_folder(runs, run["cfg"]["scenario"], backend_dir)
                self._send_json({"ok": True, "run_dir": run_dir})
                return
            self._send_json({"error": "not found"}, status=404)

    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--file", default="feddrl_dashboard.html",
                    help="dashboard HTML file to open (default: feddrl_dashboard.html)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--results_dir", default="results_backend",
                    help="folder where dashboard runs are persisted (default: results_backend)")
    args = p.parse_args()

    serve_dir = os.path.dirname(os.path.abspath(args.file)) or "."
    file_name = os.path.basename(args.file)
    full_path = os.path.join(serve_dir, file_name)
    if not os.path.exists(full_path):
        print(f"ERROR: could not find '{full_path}'.")
        print("Put server.py in the same folder as the dashboard HTML file,")
        print("or pass --file /full/path/to/feddrl_dashboard.html")
        raise SystemExit(1)

    backend_dir = os.path.join(serve_dir, args.results_dir)
    os.makedirs(backend_dir, exist_ok=True)

    port = find_free_port(args.port)
    handler = make_handler(serve_dir, backend_dir)

    with socketserver.ThreadingTCPServer(("127.0.0.1", port), handler) as httpd:
        url = f"http://127.0.0.1:{port}/{file_name}"
        print(f"Serving '{file_name}' from '{serve_dir}'")
        print(f"Dashboard results persist to: {backend_dir}")
        print(f"Dashboard running at: {url}")
        print("Press Ctrl+C to stop.")
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()