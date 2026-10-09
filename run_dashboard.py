#!/usr/bin/env python3
"""
run_dashboard.py -- one-command launcher for feddrl_dashboard.html
======================================================================
Usage:
    python run_dashboard.py
    python run_dashboard.py --file feddrl_dashboard.html --port 8000 --results results

What it does:
  1. Downloads Chart.js and JSZip into this folder (only if missing).
  2. Starts a local web server and opens the dashboard in your browser.
  3. Receives finished runs from the dashboard and stores them as

        results/
            proposed_work/<run_name>_<timestamp>/   <- Federated DRL runs
            base_paper/<run_name>_<timestamp>/      <- Baseline DDQN runs
                config.json, metrics.csv, reward.png,
                handoff_probability.png, pingpong_probability.png, README.txt

     (identical contents to the dashboard's .zip export).

Keep this file in the SAME folder as feddrl_dashboard.html.
"""

import argparse
import base64
import http.server
import json
import os
import re
import socket
import socketserver
import sys
import threading
import urllib.request
import webbrowser

LIBS = {
    "chart.umd.min.js": [
        "https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js",
        "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.4/chart.umd.min.js",
    ],
    "jszip.min.js": [
        "https://cdn.jsdelivr.net/npm/jszip@3.10.1/dist/jszip.min.js",
        "https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js",
    ],
}

CATEGORIES = ("proposed_work", "base_paper")
ALLOWED_FILES = {
    "config.json", "metrics.csv", "README.txt",
    "reward.png", "handoff_probability.png", "pingpong_probability.png",
}
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]*$")
MAX_BODY = 25 * 1024 * 1024


def ensure_libs(folder: str) -> None:
    for name, urls in LIBS.items():
        dest = os.path.join(folder, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            continue
        ok = False
        for url in urls:
            try:
                print(f"Downloading {name} from {url} ...")
                with urllib.request.urlopen(url, timeout=20) as r:
                    data = r.read()
                if len(data) < 1000:
                    raise ValueError("response too small")
                with open(dest, "wb") as f:
                    f.write(data)
                ok = True
                break
            except Exception as e:
                print(f"  failed: {e}")
        if not ok:
            print(f"WARNING: could not download {name}. Download it manually from:")
            print(f"  {urls[0]}")
            print(f"and save it as '{dest}'.")


def find_free_port(preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", default="feddrl_dashboard.html")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--results", default=None,
                   help="results folder (default: <dashboard folder>/results)")
    args = p.parse_args()

    serve_dir = os.path.dirname(os.path.abspath(args.file)) or "."
    file_name = os.path.basename(args.file)
    full_path = os.path.join(serve_dir, file_name)
    results_dir = os.path.abspath(args.results or os.path.join(serve_dir, "results"))

    if not os.path.exists(full_path):
        print(f"ERROR: could not find '{full_path}'.")
        print("Put run_dashboard.py in the same folder as the dashboard HTML file,")
        print("or pass --file /full/path/to/feddrl_dashboard.html")
        sys.exit(1)

    ensure_libs(serve_dir)
    for cat in CATEGORIES:
        os.makedirs(os.path.join(results_dir, cat), exist_ok=True)
    port = find_free_port(args.port)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=serve_dir, **kw)

        def log_message(self, fmt, *a):
            pass

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != "/save":
                return self._json(404, {"ok": False, "error": "not found"})
            try:
                n = int(self.headers.get("Content-Length", 0))
                if n <= 0 or n > MAX_BODY:
                    raise ValueError("bad request size")
                data = json.loads(self.rfile.read(n))
                cat, run, files = data["category"], data["run_dir"], data["files"]
                if cat not in CATEGORIES:
                    raise ValueError("unknown category")
                if not SAFE_NAME.match(run):
                    raise ValueError("unsafe run name")
                out = os.path.join(results_dir, cat, run)
                os.makedirs(out, exist_ok=True)
                for name, content in files.items():
                    if name not in ALLOWED_FILES:
                        continue
                    path = os.path.join(out, name)
                    if "b64" in content:
                        with open(path, "wb") as f:
                            f.write(base64.b64decode(content["b64"]))
                    else:
                        with open(path, "w", encoding="utf-8", newline="") as f:
                            f.write(content["text"])
                rel = os.path.relpath(out, os.path.dirname(results_dir)).replace(os.sep, "/")
                print(f"Saved run -> {out}")
                self._json(200, {"ok": True, "path": rel})
            except Exception as e:
                self._json(400, {"ok": False, "error": str(e)})

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), Handler) as httpd:
        url = f"http://127.0.0.1:{port}/{file_name}"
        print(f"Serving '{file_name}' from '{serve_dir}'")
        print(f"Results are stored in: {results_dir}")
        print(f"Dashboard running at: {url}")
        print("Press Ctrl+C to stop.")
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()