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
from optuna import TrialPruned

sys.path.insert(0, str(Path(__file__).parent.parent))

from feature_pipeline import load_and_engineer
from models import run_trial
from search_space import sample_config, sample_config_phase1, sample_config_phase2


def make_objective(
    data_path: str,
    budget_seconds: float,
    results_dir: Path,
    tag: str,
    families: list[str] | None = None,
    phase: int | None = None,
    phase2_family: str | None = None,
    phase2_params: dict | None = None,
) -> Any:
    """Return an Optuna objective with pre-loaded data.

    The DataFrame is engineered once and shared across all trials in the
    process.  Each trial selects a feature subset and model family, trains
    within ``budget_seconds``, and logs to TSV + JSONL.
    
    Parameters
    ----------
    phase : int | None
        If None, joint search (sample family + HPs + features together).
        If 1, phase 1: sample family + HPs, fix features to all available.
        If 2, phase 2: fix family + HPs, sample feature subsets only.
          For phase 2, must provide phase2_family and phase2_params.
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
    timing_path = results_dir / f"timing-{tag}.log"

    if not tsv_path.exists():
        tsv_path.write_text("trial\trmse\tstatus\tdescription\n")

    # Mutable best tracker shared across all trials in this session.
    _best: dict = {"rmse": float("inf"), "trial": None}

    def _log(
        trial_num: int,
        rmse: float,
        status: str,
        desc: str,
        params: dict,
        best_rmse_before: float,
    ) -> None:
        with tsv_path.open("a") as f:
            f.write(f"{trial_num}\t{rmse:.6f}\t{status}\t{desc}\n")
        delta_pct = (
            round((rmse - best_rmse_before) / best_rmse_before * 100, 2)
            if math.isfinite(rmse) and math.isfinite(best_rmse_before)
            else None
        )
        record = {
            "trial": trial_num,
            "status": status,
            "rmse": rmse if math.isfinite(rmse) else 0.0,
            "best_rmse_before": best_rmse_before if math.isfinite(best_rmse_before) else None,
            "delta_pct": delta_pct,
            "description": desc,
            "params": params,
        }
        with jsonl_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def objective(trial: optuna.Trial) -> float:
        if phase == 1:
            config = sample_config_phase1(trial, families=families)
        elif phase == 2:
            if phase2_family is None or phase2_params is None:
                raise ValueError(
                    "Phase 2 requires phase2_family and phase2_params to be set"
                )
            config = sample_config_phase2(trial, phase2_family, phase2_params)
        else:
            # Default: joint search
            config = sample_config(trial, families=families)

        family = config["family"]
        feat_key = "seq_features" if family in ("cnn", "gru") else "features"
        features_str = ",".join(config.get(feat_key, []))
        desc = f"{family} | {features_str[:80]}"

        best_rmse_before = _best["rmse"]
        print(f"\n[trial {trial.number}] {desc}")
        
        # Log trial start
        from datetime import datetime, timezone
        start_time = datetime.now(timezone.utc).isoformat()
        with timing_path.open("a") as f:
            f.write(f"trial {trial.number} start {start_time}\n")
        
        t_start = time.time()
        try:
            # For neural families with trial/pruning, run in-process to avoid pickling trial.
            # For others, use ProcessPoolExecutor for timeout protection.
            family = config["family"]
            if family in ("cnn", "gru"):
                # Run in-process; pruning check will happen inside the trainer
                try:
                    rmse = run_trial(config, df, budget_seconds, trial)
                except TrialPruned:
                    elapsed = time.time() - t_start
                    print(f"  PRUNED after {elapsed:.0f}s")
                    end_time = datetime.now(timezone.utc).isoformat()
                    with timing_path.open("a") as f:
                        f.write(f"trial {trial.number} end {end_time} pruned\n")
                    _log(
                        trial.number,
                        float("nan"),
                        "pruned",
                        desc,
                        trial.params,
                        best_rmse_before,
                    )
                    # Returning inf for pruned trials means Optuna skips them in results
                    return float("inf")
            else:
                # Tabular models: use ProcessPoolExecutor for timeout protection
                with ProcessPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(run_trial, config, df, budget_seconds, None)
                    try:
                        rmse = future.result(timeout=budget_seconds)
                    except _TimeoutError:
                        future.cancel()
                        raise RuntimeError(f"trial timed out after {budget_seconds:.0f}s")

            if not math.isfinite(rmse):
                raise ValueError(f"non-finite RMSE: {rmse}")
            elapsed = time.time() - t_start
            if rmse < _best["rmse"]:
                _best["rmse"] = rmse
                _best["trial"] = trial.number
                status = "keep"
                print(f"  rmse={rmse:.6f}  elapsed={elapsed:.0f}s  ** new best **")
            else:
                status = "discard"
                print(f"  rmse={rmse:.6f}  elapsed={elapsed:.0f}s")
            
            end_time = datetime.now(timezone.utc).isoformat()
            with timing_path.open("a") as f:
                f.write(f"trial {trial.number} end {end_time} {status}\n")
            
            _log(trial.number, rmse, status, desc, trial.params, best_rmse_before)
            return rmse
        except Exception as exc:
            elapsed = time.time() - t_start
            err = str(exc)
            if "terminated before any iteration completed" in err or "terminated before any epoch completed" in err:
                status = "terminated"
                print(f"  TERMINATED after {elapsed:.0f}s: {err}")
            elif "non-finite" in err:
                status = "numeric_fail"
                print(f"  NUMERIC_FAIL after {elapsed:.0f}s: {err}")
            else:
                status = "crash"
                print(f"  CRASH after {elapsed:.0f}s: {exc}")
            end_time = datetime.now(timezone.utc).isoformat()
            with timing_path.open("a") as f:
                f.write(f"trial {trial.number} end {end_time} {status}\n")
            _log(trial.number, 0.0, status, desc, trial.params, best_rmse_before)
            return float("inf")

    return objective
