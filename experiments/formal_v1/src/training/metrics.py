"""Flat-file metric logging for the research harness.

Two tiers of metrics, both committed to git:

  * Per-step metrics → `metrics.jsonl`
    One JSON object per line, appended every `step_log_interval` steps.
    Append-only (crash-safe), grep-able, `pd.read_json(lines=True)`-ready.
    Used for training curves (loss, lr, grad-norm, throughput).

  * Per-epoch metrics → `history.csv`
    One row per epoch, written atomically (tmp + rename).
    Used for thesis tables and cross-run aggregation.

No TensorBoard. Live debugging is done via `tail -f history.csv` and
console summaries from the run script.
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass
class MetricLogger:
    """Writes per-step JSONL and per-epoch CSV artifacts to a run directory.

    Usage:
        logger = MetricLogger(run_dir, step_log_interval=50)
        logger.log_step({"train_loss": 2.3, "lr": 3e-4, ...})
        logger.log_epoch({"epoch": 1, "dev_cer": 0.47, ...})
        logger.close()
    """

    run_dir: str | Path
    step_log_interval: int = 50

    _step: int = field(default=0, init=False)
    _epoch_header_written: bool = field(default=False, init=False)
    _history_path: Path = field(init=False)
    _metrics_path: Path = field(init=False)
    _metrics_fp: Any = field(default=None, init=False)
    _t_last_flush: float = field(default_factory=time.time, init=False)

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._history_path = self.run_dir / "history.csv"
        self._metrics_path = self.run_dir / "metrics.jsonl"
        # Open in append mode so resume-on-crash extends the file.
        self._metrics_fp = open(self._metrics_path, "a", buffering=1)
        # Detect whether history.csv already has a header (resume case).
        self._epoch_header_written = (
            self._history_path.exists() and self._history_path.stat().st_size > 0
        )

    # ── per-step ──────────────────────────────────────────────────────────
    def log_step(self, payload: Mapping[str, Any]) -> None:
        """Record one step's metrics. Only flushed every `step_log_interval` steps.

        The caller is expected to pass `step` and `epoch` inside `payload`.
        """
        self._step += 1
        if self._step % self.step_log_interval != 0:
            return
        json.dump({k: _to_jsonable(v) for k, v in payload.items()}, self._metrics_fp)
        self._metrics_fp.write("\n")
        self._metrics_fp.flush()

    def force_log_step(self, payload: Mapping[str, Any]) -> None:
        """Record regardless of interval (useful at epoch boundaries)."""
        json.dump({k: _to_jsonable(v) for k, v in payload.items()}, self._metrics_fp)
        self._metrics_fp.write("\n")
        self._metrics_fp.flush()

    # ── per-epoch ─────────────────────────────────────────────────────────
    def log_epoch(self, row: Mapping[str, Any]) -> None:
        """Append one row to `history.csv`. Writes header on first call."""
        row = {k: _to_jsonable(v) for k, v in row.items()}
        tmp_path = self._history_path.with_suffix(".csv.tmp")

        # Read existing rows (cheap: O(epochs) ≤ 80).
        existing: list[dict] = []
        if self._epoch_header_written:
            with open(self._history_path, newline="") as f:
                existing = list(csv.DictReader(f))

        # Union of keys preserves column order across epochs.
        fieldnames: list[str] = []
        for prior in existing + [row]:
            for k in prior.keys():
                if k not in fieldnames:
                    fieldnames.append(k)

        with open(tmp_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for prior in existing:
                writer.writerow({k: prior.get(k, "") for k in fieldnames})
            writer.writerow({k: row.get(k, "") for k in fieldnames})

        os.replace(tmp_path, self._history_path)
        self._epoch_header_written = True

    def close(self) -> None:
        if self._metrics_fp is not None and not self._metrics_fp.closed:
            self._metrics_fp.close()

    def __enter__(self) -> "MetricLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def _to_jsonable(v: Any) -> Any:
    """Convert torch tensors / numpy scalars to plain Python types."""
    if hasattr(v, "item") and callable(v.item):
        try:
            return v.item()
        except Exception:
            return str(v)
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_to_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _to_jsonable(x) for k, x in v.items()}
    return str(v)
