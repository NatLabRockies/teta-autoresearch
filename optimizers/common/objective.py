"""Generic Optuna objective factory.

`make_objective()` pre-loads the dataset via the domain's `load_and_engineer`,
then returns a closure that Optuna calls per trial. Each trial samples a
config via the domain's `sample_config*` entry point, runs
`domain.run_trial(...)`, and logs through `TrialLogger`.
"""

from __future__ import annotations

import math
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as _TimeoutError
from pathlib import Path
from typing import Any, Callable

import optuna
from optuna import TrialPruned

from .domain_loader import DomainHooks
from .logging import TrialLogger

# Families that train neural nets and want in-process execution (so the
# Optuna Trial object can be used for pruning). Any family not in this set
# is wrapped in a ProcessPoolExecutor with a hard wall-clock timeout.
_IN_PROCESS_FAMILIES = {"cnn", "gru"}


def _pick_sampler(
    domain: DomainHooks,
    phase: int | None,
    phase2_family: str | None,
    phase2_params: dict | None,
    families: list[str] | None,
) -> Callable[[optuna.Trial], dict]:
    if phase == 1:
        return lambda t: domain.module.sample_config_phase1(t, families=families)
    if phase == 2:
        if phase2_family is None or phase2_params is None:
            raise ValueError(
                "phase 2 requires phase2_family and phase2_params to be set"
            )
        return lambda t: domain.module.sample_config_phase2(
            t, phase2_family, phase2_params
        )
    return lambda t: domain.sample_config(t, families=families)


def make_objective(
    domain: DomainHooks,
    data_path: str,
    budget_seconds: float,
    logger: TrialLogger,
    families: list[str] | None = None,
    phase: int | None = None,
    phase2_family: str | None = None,
    phase2_params: dict | None = None,
) -> Callable[[optuna.Trial], float]:
    data_path_abs = Path(data_path).resolve()
    if not data_path_abs.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path_abs}\n"
            "Pass --data-path explicitly or --partition <val> to let the "
            "domain resolve a default path."
        )

    print(f"Loading and engineering features from {data_path_abs} …")
    t0 = time.time()
    df = domain.load_and_engineer(str(data_path_abs))
    print(f"Data ready in {time.time() - t0:.1f}s  ({len(df):,} rows)")

    sampler = _pick_sampler(domain, phase, phase2_family, phase2_params, families)
    run_trial = domain.run_trial

    best: dict[str, Any] = {"rmse": float("inf"), "trial": None}

    def objective(trial: optuna.Trial) -> float:
        config = sampler(trial)
        family = config["family"]
        feat_key = "seq_features" if family in _IN_PROCESS_FAMILIES else "features"
        features_str = ",".join(config.get(feat_key, []))
        desc = f"{family} | {features_str[:80]}"

        best_rmse_before = best["rmse"]
        print(f"\n[trial {trial.number}] {desc}")
        logger.start(trial.number)

        t_start = time.time()
        try:
            if family in _IN_PROCESS_FAMILIES:
                try:
                    rmse = run_trial(config, df, budget_seconds, trial)
                except TrialPruned:
                    elapsed = time.time() - t_start
                    print(f"  PRUNED after {elapsed:.0f}s")
                    logger.end(trial.number, "pruned")
                    logger.log(
                        trial.number,
                        float("nan"),
                        "pruned",
                        desc,
                        trial.params,
                        best_rmse_before,
                    )
                    return float("inf")
            else:
                with ProcessPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        run_trial, config, df, budget_seconds, None
                    )
                    try:
                        rmse = future.result(timeout=budget_seconds)
                    except _TimeoutError:
                        future.cancel()
                        raise RuntimeError(
                            f"trial timed out after {budget_seconds:.0f}s"
                        )

            if not math.isfinite(rmse):
                raise ValueError(f"non-finite RMSE: {rmse}")
            elapsed = time.time() - t_start
            if rmse < best["rmse"]:
                best["rmse"] = rmse
                best["trial"] = trial.number
                status = "keep"
                print(f"  rmse={rmse:.6f}  elapsed={elapsed:.0f}s  ** new best **")
            else:
                status = "discard"
                print(f"  rmse={rmse:.6f}  elapsed={elapsed:.0f}s")

            logger.end(trial.number, status)
            logger.log(trial.number, rmse, status, desc, trial.params, best_rmse_before)
            return rmse
        except Exception as exc:
            elapsed = time.time() - t_start
            err = str(exc)
            if (
                "terminated before any iteration completed" in err
                or "terminated before any epoch completed" in err
            ):
                status = "terminated"
                print(f"  TERMINATED after {elapsed:.0f}s: {err}")
            elif "non-finite" in err:
                status = "numeric_fail"
                print(f"  NUMERIC_FAIL after {elapsed:.0f}s: {err}")
            else:
                status = "crash"
                print(f"  CRASH after {elapsed:.0f}s: {exc}")
            logger.end(trial.number, status)
            logger.log(trial.number, 0.0, status, desc, trial.params, best_rmse_before)
            return float("inf")

    return objective
