"""Local HTTP server for the Foundry UI.

Wires the existing pipeline to a browser front end. Adds no dependencies:
standard library only, so it keeps working with the network off.

Nothing in this file modifies featurise.py, train.py, generate.py, spec.py,
emit.py, evaluate.py, baseline.py or run.py. Builds are run by launching
run.py as a subprocess and streaming its stdout, so the pipeline the UI
drives is byte-identical to the one you run from the terminal.

    python3 server.py            # then open http://127.0.0.1:8000
"""

import importlib.util
import json
import math
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, unquote

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
UI_FILE = os.path.join(ROOT, "ui.html")
PORT = int(os.environ.get("FOUNDRY_PORT", "8000"))

# Casting modules are cached by path and mtime: re-emitting a task picks up
# the new weights without a server restart.
_casting_cache = {}
_casting_lock = threading.Lock()


# --- loading ---------------------------------------------------------------

def read_json(path):
    with open(path) as f:
        return json.load(f)


def task_slugs():
    if not os.path.isdir(DATA_DIR):
        return []
    out = []
    for name in sorted(os.listdir(DATA_DIR)):
        if name.startswith((".", "_")):
            continue
        if os.path.exists(os.path.join(DATA_DIR, name, "spec.json")):
            out.append(name)
    return out


def casting_file(slug):
    d = os.path.join(DATA_DIR, slug)
    spec = read_json(os.path.join(d, "spec.json"))
    return os.path.join(d, f"{spec['task_name']}_classifier.py")


def load_casting(slug):
    """Import an emitted casting. Cached until the file changes on disk."""
    path = casting_file(slug)
    if not os.path.exists(path):
        raise FileNotFoundError(f"no casting for {slug}; run a build first")
    stamp = os.path.getmtime(path)
    with _casting_lock:
        hit = _casting_cache.get(path)
        if hit and hit[0] == stamp:
            return hit[1]
        spec = importlib.util.spec_from_file_location(f"casting_{slug}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _casting_cache[path] = (stamp, module)
        return module


def softmax(values):
    """Stable softmax. The casting exposes logits(); probabilities are ours."""
    top = max(values)
    exps = [math.exp(v - top) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


# --- metrics ---------------------------------------------------------------

def evaluate_set(slug, examples):
    """Score the casting over {text, label} pairs. label may be index or name."""
    spec = read_json(os.path.join(DATA_DIR, slug, "spec.json"))
    names = [l["name"] for l in spec["labels"]]
    index_of = {n: i for i, n in enumerate(names)}
    module = load_casting(slug)
    k = len(names)

    matrix = [[0] * k for _ in range(k)]
    latencies = []
    misses = []

    for ex in examples:
        gold = ex["label"]
        gold = gold if isinstance(gold, int) else index_of[gold]
        start = time.perf_counter()
        predicted = module.classify(ex["text"])
        latencies.append((time.perf_counter() - start) * 1000.0)
        matrix[gold][index_of[predicted]] += 1
        if index_of[predicted] != gold:
            misses.append({
                "text": ex["text"],
                "gold": names[gold],
                "predicted": predicted,
            })

    precision, recall, f1, support = [], [], [], []
    for c in range(k):
        tp = matrix[c][c]
        pred_c = sum(matrix[r][c] for r in range(k))
        act_c = sum(matrix[c])
        p = tp / pred_c if pred_c else 0.0
        r = tp / act_c if act_c else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(2 * p * r / (p + r) if (p + r) else 0.0)
        support.append(act_c)

    total = len(examples)
    correct = sum(matrix[c][c] for c in range(k))
    return {
        "n": total,
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(f1) / k if k else 0.0,
        "labels": names,
        "precision": precision,
        "recall": recall,
        "support": support,
        "confusion": matrix,
        "median_latency_ms": statistics.median(latencies) if latencies else 0.0,
        "misses": misses,
    }


def task_summary(slug):
    d = os.path.join(DATA_DIR, slug)
    spec = read_json(os.path.join(d, "spec.json"))
    path = casting_file(slug)
    has_casting = os.path.exists(path)
    train_path = os.path.join(d, "train_A.json")
    test_path = os.path.join(d, "test_B.json")
    return {
        "slug": slug,
        "task_name": spec["task_name"],
        "num_classes": spec["num_classes"],
        "labels": [l["name"] for l in spec["labels"]],
        "definitions": [l.get("definition", "") for l in spec["labels"]],
        "has_casting": has_casting,
        "casting_kb": round(os.path.getsize(path) / 1024, 1) if has_casting else None,
        "n_train": len(read_json(train_path)) if os.path.exists(train_path) else 0,
        "n_test": len(read_json(test_path)) if os.path.exists(test_path) else 0,
    }


def wild_files():
    return sorted(
        f for f in os.listdir(ROOT)
        if f.startswith("wild") and f.endswith(".json")
    )


# --- build (subprocess, streamed) ------------------------------------------

STAGE_PATTERN = re.compile(r"^\[(\d)/5\]")


def stream_build(need, n_train, n_test, emit_event):
    """Run run.py and forward its stdout as events. Never modifies run.py."""
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [
        sys.executable, "-u", os.path.join(ROOT, "run.py"),
        "--need", need,
        "--n-train", str(n_train),
        "--n-test", str(n_test),
    ]
    emit_event("log", {"line": "$ " + " ".join(cmd[2:])})
    started = time.time()

    process = subprocess.Popen(
        cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    task_dir = None
    for line in process.stdout:
        line = line.rstrip("\n")
        emit_event("log", {"line": line})
        stage = STAGE_PATTERN.match(line.strip())
        if stage:
            emit_event("stage", {"index": int(stage.group(1))})
        if "task dir:" in line:
            task_dir = line.split("task dir:")[1].strip()

    code = process.wait()
    elapsed = round(time.time() - started, 1)
    if code != 0:
        emit_event("failed", {"code": code, "seconds": elapsed})
        return
    slug = os.path.basename(task_dir) if task_dir else None
    emit_event("done", {"slug": slug, "seconds": elapsed})


# --- http ------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass  # the build log is the interesting output, not access lines

    def _send(self, code, payload, content_type="application/json"):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _fail(self, code, message):
        self._send(code, {"error": message})

    def do_GET(self):
        route = urlparse(self.path)
        path = unquote(route.path)
        try:
            if path in ("/", "/index.html"):
                with open(UI_FILE, "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")

            if path == "/api/tasks":
                return self._send(200, {
                    "tasks": [task_summary(s) for s in task_slugs()],
                    "wild_files": wild_files(),
                })

            if path.startswith("/api/task/"):
                slug = path[len("/api/task/"):]
                if slug not in task_slugs():
                    return self._fail(404, f"unknown task {slug}")
                summary = task_summary(slug)
                if summary["has_casting"]:
                    test = read_json(os.path.join(DATA_DIR, slug, "test_B.json"))
                    summary["test_metrics"] = evaluate_set(slug, test)
                return self._send(200, summary)

            if path.startswith("/api/wild/"):
                rest = path[len("/api/wild/"):]
                slug, _, filename = rest.partition("/")
                if slug not in task_slugs():
                    return self._fail(404, f"unknown task {slug}")
                if filename not in wild_files():
                    return self._fail(404, f"unknown wild file {filename}")
                examples = read_json(os.path.join(ROOT, filename))
                names = task_summary(slug)["labels"]
                unknown = sorted({
                    e["label"] for e in examples
                    if isinstance(e["label"], str) and e["label"] not in names
                })
                if unknown:
                    return self._fail(400, (
                        f"{filename} uses labels not in this task's spec: "
                        f"{', '.join(unknown)}"
                    ))
                result = evaluate_set(slug, examples)
                result["file"] = filename
                return self._send(200, result)

            return self._fail(404, f"no route for {path}")
        except FileNotFoundError as exc:
            return self._fail(404, str(exc))
        except Exception as exc:  # surfaced in the UI, never silently swallowed
            return self._fail(500, f"{type(exc).__name__}: {exc}")

    def do_POST(self):
        route = urlparse(self.path)
        path = unquote(route.path)
        length = int(self.headers.get("Content-Length", "0"))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as exc:
            return self._fail(400, f"body is not valid JSON: {exc}")

        try:
            if path == "/api/classify":
                slug = body.get("task")
                text = (body.get("text") or "").strip()
                if slug not in task_slugs():
                    return self._fail(404, f"unknown task {slug}")
                if len(" ".join(text.lower().split())) < 3:
                    return self._fail(400, "needs at least 3 characters")
                module = load_casting(slug)
                names = task_summary(slug)["labels"]
                start = time.perf_counter()
                raw = module.logits(text)
                elapsed = (time.perf_counter() - start) * 1000.0
                probs = softmax(raw)
                ranked = sorted(
                    ({"label": n, "p": p} for n, p in zip(names, probs)),
                    key=lambda r: -r["p"],
                )
                return self._send(200, {
                    "label": ranked[0]["label"],
                    "confidence": ranked[0]["p"],
                    "ranked": ranked,
                    "latency_ms": elapsed,
                })

            if path == "/api/build":
                return self._stream_build(body)

            return self._fail(404, f"no route for {path}")
        except FileNotFoundError as exc:
            return self._fail(404, str(exc))
        except Exception as exc:
            return self._fail(500, f"{type(exc).__name__}: {exc}")

    def _stream_build(self, body):
        need = (body.get("need") or "").strip()
        if len(need) < 10:
            return self._fail(400, "describe the task in a sentence or more")
        try:
            n_train = int(body.get("n_train", 4000))
            n_test = int(body.get("n_test", 100))
        except (TypeError, ValueError):
            return self._fail(400, "n_train and n_test must be whole numbers")

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()

        def emit_event(kind, payload):
            chunk = f"event: {kind}\ndata: {json.dumps(payload)}\n\n"
            self.wfile.write(chunk.encode())
            self.wfile.flush()

        try:
            stream_build(need, n_train, n_test, emit_event)
        except BrokenPipeError:
            pass  # the browser navigated away mid-build
        except Exception as exc:
            try:
                emit_event("failed", {"error": f"{type(exc).__name__}: {exc}"})
            except BrokenPipeError:
                pass


def main():
    if not os.path.exists(UI_FILE):
        raise SystemExit(f"ui.html not found next to server.py at {UI_FILE}")
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Foundry UI  ->  http://127.0.0.1:{PORT}")
    print(f"tasks: {', '.join(task_slugs()) or 'none yet'}")
    print("ctrl-c to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()