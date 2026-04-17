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
import json
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
        default=600.0,
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
        "--families",
        default=None,
        help=(
            "Comma-delimited list of model families to search "
            f"(choices: {', '.join(__import__('search_space').ALL_FAMILIES)}). "
            "Defaults to all families."
        ),
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
    p.add_argument(
        "--phase",
        type=int,
        choices=[1, 2],
        default=None,
        help=(
            "Phase of two-phase search: 1=tune family/HPs with fixed features, "
            "2=ablate features with fixed family/HPs. Omit for joint search."
        ),
    )
    p.add_argument(
        "--phase2-family",
        default=None,
        help=(
            "For phase 2 only: the winning model family from phase 1 "
            "(e.g. 'rf' or 'cnn')."
        ),
    )
    p.add_argument(
        "--phase2-params",
        default=None,
        help=(
            "For phase 2 only: JSON string of fixed model params from phase 1 "
            "(e.g. '{\"rf_n_estimators\": 1200, ...}'). "
            "Extract from phase 1 results or use --extract-phase1."
        ),
    )
    p.add_argument(
        "--extract-phase1",
        type=Path,
        default=None,
        help=(
            "Path to phase 1 results DB. Extract best trial params "
            "and print them (for phase 2 command)."
        ),
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

    # If --extract-phase1 is provided, load that study and print best params, then exit
    if args.extract_phase1:
        _extract_and_print_phase1(args.extract_phase1)
        return

    results_dir: Path = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    print("CONFIGURATION:")
    for k, v in vars(args).items():
        if k != "extract_phase1":  # Skip this for clarity
            print(f"  {k}: {v}")

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

    families: list[str] | None = None
    if args.families:
        from search_space import ALL_FAMILIES
        families = [f.strip() for f in args.families.split(",")]
        invalid = [f for f in families if f not in ALL_FAMILIES]
        if invalid:
            print(
                f"Error: unknown families: {invalid}. "
                f"Valid choices: {ALL_FAMILIES}",
                file=sys.stderr,
            )
            sys.exit(1)

    # Parse phase 2 params if provided
    phase2_params: dict | None = None
    if args.phase == 2:
        if not args.phase2_family:
            print("Error: phase 2 requires --phase2-family", file=sys.stderr)
            sys.exit(1)
        if not args.phase2_params:
            print("Error: phase 2 requires --phase2-params", file=sys.stderr)
            sys.exit(1)
        try:
            phase2_params = json.loads(args.phase2_params)
        except json.JSONDecodeError as e:
            print(f"Error parsing --phase2-params JSON: {e}", file=sys.stderr)
            sys.exit(1)

    # Enqueue warm-start configs as the very first trials, filtered to active families
    if not args.no_warm_start and completed == 0 and args.phase != 2:
        warm = [
            cfg for cfg in WARM_START_CONFIGS
            if families is None or cfg.get("family") in families
        ]
        print(f"Enqueuing {len(warm)} warm-start configs from learnings.md …")
        for cfg in warm:
            study.enqueue_trial(cfg)

    objective = make_objective(
        data_path=args.data_path,
        budget_seconds=args.budget,
        results_dir=results_dir,
        tag=args.tag,
        families=families,
        phase=args.phase,
        phase2_family=args.phase2_family,
        phase2_params=phase2_params,
    )

    phase_str = f"phase {args.phase}" if args.phase else "joint"
    print(
        f"\nStudy '{args.tag}'  ({phase_str})  sampler={args.sampler}  "
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
    family = best.params.get("family", "unknown")
    print(f"  family : {family}")
    print(f"  params :")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    # For phase 1 results, print extraction command for phase 2
    if family in ["rf", "extra_trees", "hgbr", "xgb", "lgbm", "mlp"]:
        print(f"\nFor phase 2, use: --phase2-family {family}")
        # Build the fixed params dict (HP keys only, not feature flags)
        hp_keys = [k for k in best.params.keys() if k != "family" and not k.startswith("feat_")]
        fixed_params = {k: best.params[k] for k in hp_keys}
        params_json = json.dumps(fixed_params)
        print(f"  --phase2-params '{params_json}'")


def _extract_and_print_phase1(db_path: Path) -> None:
    """Load a phase 1 study and print best trial params for phase 2 usage."""
    if not db_path.exists():
        print(f"Error: {db_path} does not exist", file=sys.stderr)
        sys.exit(1)

    study_name = db_path.stem.replace("search-", "")
    storage = f"sqlite:///{db_path}"
    study = optuna.load_study(study_name=study_name, storage=storage)
    _print_best(study)


if __name__ == "__main__":
    main()
