"""TSV / JSONL / timing log writers for optimizer trials.

Matches the format used by the LLM experiment sessions (see program.md):
each trial appends one row to the TSV and one JSON line to the JSONL.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class TrialLogger:
    def __init__(self, results_dir: Path, tag: str, method: str) -> None:
        results_dir.mkdir(parents=True, exist_ok=True)
        self.tsv_path = results_dir / f"search-{tag}-{method}.tsv"
        self.jsonl_path = results_dir / f"search-{tag}-{method}.jsonl"
        self.timing_path = results_dir / f"timing-{tag}-{method}.log"
        if not self.tsv_path.exists():
            self.tsv_path.write_text("trial\trmse\tstatus\tdescription\n")

    def start(self, trial_num: int) -> None:
        stamp = datetime.now(timezone.utc).isoformat()
        with self.timing_path.open("a") as f:
            f.write(f"trial {trial_num} start {stamp}\n")

    def end(self, trial_num: int, status: str) -> None:
        stamp = datetime.now(timezone.utc).isoformat()
        with self.timing_path.open("a") as f:
            f.write(f"trial {trial_num} end {stamp} {status}\n")

    def log(
        self,
        trial_num: int,
        rmse: float,
        status: str,
        desc: str,
        params: dict,
        best_rmse_before: float,
    ) -> None:
        with self.tsv_path.open("a") as f:
            f.write(f"{trial_num}\t{rmse:.6f}\t{status}\t{desc}\n")
        delta_pct: float | None
        if math.isfinite(rmse) and math.isfinite(best_rmse_before):
            delta_pct = round((rmse - best_rmse_before) / best_rmse_before * 100, 2)
        else:
            delta_pct = None
        record: dict[str, Any] = {
            "trial": trial_num,
            "status": status,
            "rmse": rmse if math.isfinite(rmse) else 0.0,
            "best_rmse_before": (
                best_rmse_before if math.isfinite(best_rmse_before) else None
            ),
            "delta_pct": delta_pct,
            "description": desc,
            "params": params,
        }
        with self.jsonl_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
