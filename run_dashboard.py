"""
Run this file to open the FedDRL dashboard in your browser.

    python run_dashboard.py

It starts a small local HTTP server (needed because some browsers block
JS features when opening an .html file directly via file://) and opens
your default browser to it automatically. Leave the terminal running;
press Ctrl+C to stop the server when you're done.

Place this file in the SAME folder as feddrl_dashboard.html (or pass
--file to point at a different path).
"""

import argparse
import http.server
import socketserver
import threading
import webbrowser
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="feddrl_dashboard.html",
                         help="dashboard HTML file to serve (default: feddrl_dashboard.html "
                              "in the same folder as this script)")
    parser.add_argument("--port", type=int, default=8000,
                         help="local port to serve on (default: 8000)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dashboard_path = args.file if os.path.isabs(args.file) else os.path.join(script_dir, args.file)

    if not os.path.isfile(dashboard_path):
        print(f"ERROR: could not find '{dashboard_path}'.")
        print("Make sure feddrl_dashboard.html is in the same folder as this "
              "script, or pass --file /path/to/feddrl_dashboard.html")
        sys.exit(1)

    serve_dir = os.path.dirname(dashboard_path)
    file_name = os.path.basename(dashboard_path)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=serve_dir, **kw)
        def log_message(self, *a):
            pass  # keep the terminal quiet

    port = args.port
    httpd = None
    for attempt in range(10):  # find a free port if the default is taken
        try:
            httpd = socketserver.TCPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            port += 1
    if httpd is None:
        print("Could not find a free port. Try --port <number>.")
        sys.exit(1)

    url = f"http://127.0.0.1:{port}/{file_name}"
    print(f"Serving dashboard at {url}")
    print("Press Ctrl+C to stop.")

    threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
        httpd.shutdown()


if __name__ == "__main__":
    main()