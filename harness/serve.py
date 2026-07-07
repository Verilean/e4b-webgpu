#!/usr/bin/env python3
"""Dev server for the e4b-webgpu engine.

- Serves the repo root (index.html, kernels/*.wgsl, model/*.safetensors, goldens/).
- HTTP Range support (the loader range-fetches slices of model.safetensors).
- POST /log appends to harness/run.log (the engine's println channel);
  POST /result writes harness/result.json (structured test output).
Port 8877.
"""
import http.server
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
LOG = os.path.join(ROOT, "harness", "run.log")

class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # never cache code: Chrome's module cache + same-second mtimes caused
        # silent stale-JS runs (a lost-edit bug cost a debugging hour)
        if self.path.split("?")[0].endswith((".js", ".html", ".wgsl")):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    protocol_version = "HTTP/1.1"

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        if self.path == "/result":
            with open(os.path.join(ROOT, "harness", "result.json"), "wb") as f:
                f.write(body)
        else:
            with open(LOG, "ab") as f:
                f.write(body + b"\n")
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def send_head(self):
        """SimpleHTTPRequestHandler with single-range support (bytes=a-b)."""
        path = self.translate_path(self.path)
        rng = self.headers.get("Range")
        if rng and os.path.isfile(path):
            m = re.match(r"bytes=(\d+)-(\d*)", rng)
            if m:
                size = os.path.getsize(path)
                a = int(m.group(1))
                b = int(m.group(2)) if m.group(2) else size - 1
                b = min(b, size - 1)
                f = open(path, "rb")
                f.seek(a)
                self.send_response(206)
                self.send_header("Content-Type", self.guess_type(path))
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
                self.send_header("Content-Length", str(b - a + 1))
                self.end_headers()
                self._range_len = b - a + 1
                return f
        return super().send_head()

    def copyfile(self, source, outputfile):
        n = getattr(self, "_range_len", None)
        self._range_len = None
        if n is None:
            return super().copyfile(source, outputfile)
        remaining = n
        while remaining > 0:
            chunk = source.read(min(1 << 20, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)

    def log_message(self, fmt, *args):
        pass

print(f"serving {ROOT} on :8877", flush=True)
http.server.ThreadingHTTPServer(("127.0.0.1", 8877), Handler).serve_forever()
