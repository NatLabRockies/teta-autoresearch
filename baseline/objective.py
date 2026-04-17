"""Optuna objective factory.

``make_objective()`` pre-loads and engineers the dataset once, then returns
a closure that Optuna calls for each trial.  Results are appended to both
a TSV summary and a JSONL structured log in the same format as the LLM
experiment session (program.md).
"""

import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as _TimeoutError
from pathlib import Path
from typing import Any

import optuna

sys.path.insert(0, str(Path(__file__).parent.parent))

from feature_pipeline import load_and_engineer
from models import run_trial
from search_space import sample_config


def make_objective(
    data_path: str,
    budget_seconds: float,
    results_dir: Path,
    tag: str,
    families: list[str] | None = None,
) -> Any:
    """Return an Optuna objective with pre-loaded data.

    The DataFrame is engineered once and shared across all trials in the
    process.  Each trial selects a feature subset and model family, trains
    within ``budget_seconds``, and logs to TSV + JSONL.
    """
    data_path_abs = Path(data_path).resolve()
    if not data_path_abs.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path_abs}\n"
            "Pass --data-path pointing to the processed parquet."
        )

    print(f"Loading and engineering features from {data_path_abs} …")
    t0 = time.time()
    df = load_and_engineer(str(data_path_abs))
    print(f"Data ready in {time.time() - t0:.1f}s  ({len(df):,} rows)")

    results_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = results_dir / f"search-{tag}.tsv"
    jsonl_path = results_dir / f"search-{tag}.jsonl"

    if not tsv_path.exists():
        tsv_path.write_text("trial\trmse\tstatus\tdescription\n")

    def _log(trial_num: int, rmse: float, status: str, desc: str, params: dict) -> None:
        with tsv_path.open("a") as f:
            f.write(f"{trial_num}\t{rmse:.6f}\t{status}\t{desc}\n")
        record = {
            "trial": trial_num,
            "status": status,
            "rmse": rmse,
            "description": desc,
            "params": params,
        }
        with jsonl_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def objective(trial: optuna.Trial) -> float:
        config = sample_config(trial, families=families)
        family = config["family"]
        feat_key = "seq_features" if family in ("cnn", "gru") else "features"
        features_str = ",".join(config.get(feat_key, []))
        desc = f"{family} | {features_str[:80]}"

        print(f"\n[trial {trial.number}] {desc}")
        t_start = time.time()
        try:
            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_trial, config, df, budget_seconds)
                try:
                    rmse = future.result(timeout=budget_seconds)
                except _TimeoutError:
                    future.cancel()
                    raise RuntimeError(f"trial timed out after {budget_seconds:.0f}s")
            if not math.isfinite(rmse):
                raise ValueError(f"non-finite RMSE: {rmse}")
            elapsed = time.time() - t_start
            print(f"  rmse={rmse:.6f}  elapsed={elapsed:.0f}s")
            _log(trial.number, rmse, "evaluated", desc, trial.params)
            return rmse
        except Exception as exc:
            elapsed = time.time() - t_start
            print(f"  CRASH after {elapsed:.0f}s: {exc}")
            _log(trial.number, 0.0, "crash", desc, trial.params)
            return float("inf")

    return objective
