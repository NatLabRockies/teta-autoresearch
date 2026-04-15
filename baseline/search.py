#!/usr/bin/env python3
"""Metaheuristic search over model families and hyperparameters.

Usage
-----
    pixi run python search.py --tag apr15 --n-trials 200
    pixi run python search.py --tag apr15 --n-trials 200 --sampler cmaes --budget 300
    pixi run python search.py --tag apr15 --n-trials 50  --sampler random --no-warm-start

The study is persisted to SQLite so interrupted runs can be resumed by
re-running with the same --tag.  Each trial appends one row to the TSV and
one JSON line to the JSONL, matching the format used by the LLM experiment
sessions (see ../program.md).
"""

import argparse
import sys
from pathlib import Path

import optuna

# Suppress Optuna's per-trial info logs — we print our own
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent.parent))

from objective import make_objective
from search_space import WARM_START_CONFIGS

_HERE = Path(__file__).parent
DEFAULT_DATA_PATH = str(_HERE / "../data/processed/2017_Chevy_Bolt.parquet")
DEFAULT_RESULTS_DIR = _HERE / "results"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Metaheuristic HP search for RouteE vehicle energy models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--tag",
        required=True,
        help="Run tag (e.g. apr15). Names result files and the Optuna study.",
    )
    p.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Total number of trials to run (including resumed trials).",
    )
    p.add_argument(
        "--sampler",
        choices=["tpe", "cmaes", "random"],
        default="tpe",
        help="Optuna sampler strategy.",
    )
    p.add_argument(
        "--budget",
        type=float,
        default=120.0,
        help="Per-trial training wall-clock budget in seconds.",
    )
    p.add_argument(
        "--parallel",
        type=int,
        default=1,
        help=(
            "Number of concurrent trials.  Values >1 require the SQLite "
            "storage to be shared across workers (e.g. via multiple processes "
            "each invoking search.py with the same --tag)."
        ),
    )
    p.add_argument(
        "--data-path",
        default=DEFAULT_DATA_PATH,
        help="Path to the processed parquet file.",
    )
    p.add_argument(
        "--no-warm-start",
        action="store_true",
        help="Skip enqueuing the known-good configs from learnings.md.",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory for TSV, JSONL, and SQLite outputs.",
    )
    return p.parse_args()


def _build_sampler(name: str) -> optuna.samplers.BaseSampler:
    if name == "tpe":
        return optuna.samplers.TPESampler(seed=42)
    if name == "cmaes":
        return optuna.samplers.CmaEsSampler(seed=42)
    return optuna.samplers.RandomSampler(seed=42)


def main() -> None:
    args = parse_args()
    results_dir: Path = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    storage_path = results_dir / f"search-{args.tag}.db"
    storage = f"sqlite:///{storage_path}"

    study = optuna.create_study(
        direction="minimize",
        study_name=args.tag,
        storage=storage,
        load_if_exists=True,
        sampler=_build_sampler(args.sampler),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )

    completed = len([t for t in study.trials if t.state.is_finished()])
    remaining = args.n_trials - completed
    if remaining <= 0:
        print(f"Study '{args.tag}' already has {completed} finished trials (>= {args.n_trials}). Done.")
        _print_best(study)
        return

    # Enqueue warm-start configs as the very first trials
    if not args.no_warm_start and completed == 0:
        print(f"Enqueuing {len(WARM_START_CONFIGS)} warm-start configs from learnings.md …")
        for cfg in WARM_START_CONFIGS:
            study.enqueue_trial(cfg)

    objective = make_objective(
        data_path=args.data_path,
        budget_seconds=args.budget,
        results_dir=results_dir,
        tag=args.tag,
    )

    print(
        f"\nStudy '{args.tag}'  sampler={args.sampler}  "
        f"budget={args.budget:.0f}s/trial  "
        f"trials={completed}+{remaining}={args.n_trials}"
    )
    print(f"Storage : {storage_path}")
    print(f"Results : {results_dir}/search-{args.tag}.[tsv|jsonl]")
    print()

    study.optimize(
        objective,
        n_trials=remaining,
        n_jobs=args.parallel,
        show_progress_bar=False,
    )

    _print_best(study)


def _print_best(study: optuna.Study) -> None:
    finished = [t for t in study.trials if t.state.is_finished() and t.value is not None]
    if not finished:
        print("\nNo completed trials yet.")
        return
    best = study.best_trial
    print("\n=== Best trial ===")
    print(f"  trial  : {best.number}")
    print(f"  rmse   : {best.value:.6f}")
    print(f"  family : {best.params.get('family', 'unknown')}")
    print(f"  params :")
    for k, v in best.params.items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
