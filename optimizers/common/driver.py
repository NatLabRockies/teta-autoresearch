"""Shared study driver for every sampler.

Call `run(method, sampler, args)` from each `optimizers/<method>/search.py`.
This owns the SQLite-backed Optuna study lifecycle, warm-start enqueuing,
phase 1/2 auxiliary commands, and post-run reporting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import optuna

from . import domain_loader
from .logging import TrialLogger
from .objective import make_objective

# Suppress Optuna's per-trial info logs — we print our own.
optuna.logging.set_verbosity(optuna.logging.WARNING)


def run(
    method: str, sampler: optuna.samplers.BaseSampler, args: argparse.Namespace
) -> None:
    if args.search_budget is not None and args.search_budget <= 0:
        print("Error: --search-budget must be > 0", file=sys.stderr)
        sys.exit(1)

    domain = domain_loader.load(args.domain)

    if args.extract_phase1:
        _extract_and_print_phase1(args.extract_phase1, domain)
        return
    if args.format_warm_start:
        _format_warm_start_config(args.format_warm_start)
        return

    data_path = _resolve_data_path(args, domain)

    results_dir: Path = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    print("CONFIGURATION:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print(f"  method: {method}")
    print(f"  data_path: {data_path}")

    storage_path = results_dir / f"search-{args.tag}-{method}.db"
    storage = f"sqlite:///{storage_path}"

    study = optuna.create_study(
        direction="minimize",
        study_name=f"{args.tag}-{method}",
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )

    completed = len([t for t in study.trials if t.state.is_finished()])
    remaining = args.n_trials - completed
    if remaining <= 0:
        print(
            f"Study '{study.study_name}' already has {completed} finished trials "
            f"(>= {args.n_trials}). Done."
        )
        _print_best(study)
        return

    families = _parse_families(args, domain)

    phase2_params = _parse_phase2_params(args)

    if not args.no_warm_start and completed == 0 and args.phase != 2:
        warm_configs = domain.optional("WARM_START_CONFIGS", default=[])
        warm = [
            cfg
            for cfg in warm_configs
            if families is None or cfg.get("family") in families
        ]
        if warm:
            print(f"Enqueuing {len(warm)} warm-start configs …")
            for cfg in warm:
                study.enqueue_trial(cfg)

    logger = TrialLogger(results_dir=results_dir, tag=args.tag, method=method)

    objective = make_objective(
        domain=domain,
        data_path=str(data_path),
        budget_seconds=args.budget,
        logger=logger,
        families=families,
        phase=args.phase,
        phase2_family=args.phase2_family,
        phase2_params=phase2_params,
    )

    phase_str = f"phase {args.phase}" if args.phase else "joint"
    print(
        f"\nStudy '{study.study_name}'  ({phase_str})  sampler={method}  "
        f"budget={args.budget:.0f}s/trial  "
        f"trials={completed}+{remaining}={args.n_trials}"
    )
    if args.search_budget is not None:
        print(f"Search budget : {args.search_budget:.0f}s total")
    print(f"Storage : {storage_path}")
    print(f"Results : {logger.tsv_path}  /  {logger.jsonl_path}")
    print()

    study.optimize(
        objective,
        n_trials=remaining,
        timeout=args.search_budget,
        n_jobs=args.parallel,
        show_progress_bar=False,
    )

    _print_best(study)


def _resolve_data_path(
    args: argparse.Namespace, domain: domain_loader.DomainHooks
) -> Path:
    if args.data_path:
        return Path(args.data_path)
    if args.partition is None:
        raise SystemExit(
            "Either --data-path or --partition is required (partition lets "
            "the domain resolve a default path)."
        )
    resolver = domain.optional("default_data_path")
    if resolver is None:
        raise SystemExit(
            f"domain {domain.name!r} does not expose `default_data_path`; "
            "pass --data-path explicitly."
        )
    return Path(resolver(args.partition))


def _parse_families(
    args: argparse.Namespace, domain: domain_loader.DomainHooks
) -> list[str] | None:
    if not args.families:
        return None
    all_families = domain.optional("ALL_FAMILIES")
    families = [f.strip() for f in args.families.split(",")]
    if all_families is not None:
        invalid = [f for f in families if f not in all_families]
        if invalid:
            print(
                f"Error: unknown families: {invalid}. "
                f"Valid choices for domain {domain.name!r}: {all_families}",
                file=sys.stderr,
            )
            sys.exit(1)
    return families


def _parse_phase2_params(args: argparse.Namespace) -> dict | None:
    if args.phase != 2:
        return None
    if not args.phase2_family:
        print("Error: phase 2 requires --phase2-family", file=sys.stderr)
        sys.exit(1)
    if not args.phase2_params:
        print("Error: phase 2 requires --phase2-params", file=sys.stderr)
        sys.exit(1)
    try:
        return json.loads(args.phase2_params)
    except json.JSONDecodeError as e:
        print(f"Error parsing --phase2-params JSON: {e}", file=sys.stderr)
        sys.exit(1)
    return None


def _print_best(study: optuna.Study) -> None:
    finished = [
        t for t in study.trials if t.state.is_finished() and t.value is not None
    ]
    if not finished:
        print("\nNo completed trials yet.")
        return
    best = study.best_trial
    print("\n=== Best trial ===")
    print(f"  trial  : {best.number}")
    print(f"  rmse   : {best.value:.6f}")
    family = best.params.get("family", "unknown")
    print(f"  family : {family}")
    print("  params :")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    tabular_families = ("rf", "extra_trees", "hgbr", "xgb", "lgbm", "mlp")
    if family in tabular_families:
        print(f"\nFor phase 2, use: --phase2-family {family}")
        hp_keys = [
            k
            for k in best.params.keys()
            if k != "family"
            and not k.startswith("feat_")
            and not k.startswith("seq_feat_")
            and not k.startswith("static_feat_")
        ]
        fixed_params = _phase2_fixed_params_from_best_params(
            family, best.params, hp_keys
        )
        print(f"  --phase2-params '{json.dumps(fixed_params)}'")

    print("\n=== Warm-start config (for WARM_START_CONFIGS) ===")
    config: dict[str, Any] = {"family": family}
    for k, v in best.params.items():
        if k != "family":
            config[k] = v
    _print_warm_config(config)


def _print_warm_config(config: dict[str, Any]) -> None:
    print("    {")
    for k, v in config.items():
        if isinstance(v, str):
            print(f'        "{k}": "{v}",')
        elif isinstance(v, bool):
            print(f'        "{k}": {str(v)},')
        elif isinstance(v, float):
            print(f'        "{k}": {v},')
        else:
            print(f'        "{k}": {v},')
    print("    },")


def _extract_and_print_phase1(db_path: Path, domain: domain_loader.DomainHooks) -> None:
    if not db_path.exists():
        print(f"Error: {db_path} does not exist", file=sys.stderr)
        sys.exit(1)
    study_name = db_path.stem.replace("search-", "")
    storage = f"sqlite:///{db_path}"
    study = optuna.load_study(study_name=study_name, storage=storage)
    _print_best(study)


def _phase2_fixed_params_from_best_params(
    family: str,
    best_params: dict[str, Any],
    hp_keys: list[str],
) -> dict[str, Any]:
    family_prefix = {
        "rf": "rf_",
        "extra_trees": "et_",
        "hgbr": "hgbr_",
        "xgb": "xgb_",
        "lgbm": "lgbm_",
        "mlp": "mlp_",
        "cnn": "cnn_",
        "gru": "gru_",
    }
    special_key_map = {
        ("hgbr", "lr"): "learning_rate",
        ("hgbr", "l2"): "l2_regularization",
        ("xgb", "lr"): "learning_rate",
        ("lgbm", "lr"): "learning_rate",
        ("mlp", "lr"): "learning_rate_init",
    }
    prefix = family_prefix.get(family, f"{family}_")
    fixed_params: dict[str, Any] = {}
    for key in hp_keys:
        value = best_params[key]
        base_key = key[len(prefix) :] if key.startswith(prefix) else key
        model_key = special_key_map.get((family, base_key), base_key)
        fixed_params[model_key] = value
    if family == "mlp" and "n_layers" in fixed_params and "layer_size" in fixed_params:
        n_layers = int(fixed_params.pop("n_layers"))
        layer_size = int(fixed_params.pop("layer_size"))
        fixed_params["hidden_layer_sizes"] = [layer_size] * n_layers
    return fixed_params


def _format_warm_start_config(db_path: Path) -> None:
    if not db_path.exists():
        print(f"Error: {db_path} does not exist", file=sys.stderr)
        sys.exit(1)
    study_name = db_path.stem.replace("search-", "")
    storage = f"sqlite:///{db_path}"
    study = optuna.load_study(study_name=study_name, storage=storage)
    finished = [
        t for t in study.trials if t.state.is_finished() and t.value is not None
    ]
    if not finished:
        print("\nNo completed trials yet.", file=sys.stderr)
        sys.exit(1)
    best = study.best_trial
    family = best.params.get("family", "unknown")
    config: dict[str, Any] = {"family": family}
    for k, v in best.params.items():
        if k != "family":
            config[k] = v
    print(f"\n# {study_name}: Best trial {best.number}, RMSE {best.value:.6f}")
    _print_warm_config(config)
