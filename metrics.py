"""
Backend run metrics for Living Image.

One JSON line per /api/generate run: per-frame timing and status,
failure reasons, and estimated cost. Read back through the
token-protected admin endpoints. Nothing here reaches the public API.

Railway's filesystem is ephemeral, so the file resets on redeploy.
Every run is also written to the server log as a `run_metrics {...}`
line, which Railway keeps.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import threading
from pathlib import Path

MAX_ERROR_CHARS = 300
_SECRET_ENV_KEYS = ("GEMINI_API_KEY", "OPENAI_API_KEY", "ADMIN_TOKEN")


def clean_error(text: str) -> str:
    """Strip any configured secret from an error message and bound its length."""
    out = str(text)
    for name in _SECRET_ENV_KEYS:
        value = os.environ.get(name)
        if value:
            out = out.replace(value, "[redacted]")
    return out[:MAX_ERROR_CHARS]


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100.0 * len(ordered)))
    return float(ordered[rank - 1])


def _mean_of(runs: list[dict], key: str) -> float | None:
    """Mean over runs that recorded `key`; None when none did (older runs)."""
    vals = [r[key] for r in runs if key in r]
    return statistics.fmean(vals) if vals else None


class RunMetricsStore:
    """Append-only JSONL store of run records."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._lock = threading.Lock()

    def record(self, run: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(run, separators=(",", ":"))
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def all(self) -> list[dict]:
        if not self.path.exists():
            return []
        runs = []
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        runs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return runs

    def summary(self) -> list[dict]:
        """Aggregate runs per model, in first-seen order."""
        groups: dict[str, list[dict]] = {}
        for run in self.all():
            groups.setdefault(run.get("model_id", "unknown"), []).append(run)

        rows = []
        for model_id, runs in groups.items():
            requested = sum(r.get("frames_requested", 0) for r in runs)
            failed = sum(r.get("frames_failed", 0) for r in runs)
            frame_ms = [
                f["ms"] for r in runs for f in r.get("frames", [])
                if f.get("status") == "success" and "ms" in f
            ]
            rows.append({
                "model_id": model_id,
                "model_name": runs[-1].get("model_name", model_id),
                "provider": runs[-1].get("provider", ""),
                "runs": len(runs),
                "frames_requested": requested,
                "frames_failed": failed,
                "failure_rate": (failed / requested) if requested else 0.0,
                "mean_run_ms": statistics.fmean(r.get("total_ms", 0) for r in runs),
                "median_frame_ms": statistics.median(frame_ms) if frame_ms else 0.0,
                "p95_frame_ms": _percentile(frame_ms, 95),
                "mean_cost_usd": statistics.fmean(r.get("est_cost_usd", 0) for r in runs),
                "mean_true_cost_usd": _mean_of(runs, "cost_usd"),
                "mean_input_tokens": _mean_of(runs, "input_tokens"),
                "mean_output_tokens": _mean_of(runs, "output_tokens"),
            })
        return rows
