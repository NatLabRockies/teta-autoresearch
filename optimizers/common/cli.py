"""Shared argparse for every `optimizers/<method>/search.py` entry point.

Each method's `search.py` constructs a parser via `build_parser(method)`,
fixes the sampler, and hands the parsed args to `common.driver.run(...)`.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser(method: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            f"Metaheuristic HP search (sampler={method}) — drives a domain's "
            "optimizer hooks under `domains/<name>/search/`."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--tag",
        required=True,
        help="Run tag (e.g. apr15). Names result files and the Optuna study.",
    )
    p.add_argument(
        "--domain",
        default="routee",
        help="Domain name (resolves to `domains/<name>/search`).",
    )
    p.add_argument(
        "--partition",
        default=None,
        help=(
            "Partition value within the domain (e.g. bev/ice/phev for routee). "
            "Used to resolve the default data path if --data-path is omitted."
        ),
    )
    p.add_argument(
        "--data-path",
        default=None,
        help=(
            "Path to the processed parquet. If omitted, falls back to the "
            "domain's `default_data_path(partition)` hook."
        ),
    )
    p.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Total number of trials (including resumed trials).",
    )
    p.add_argument(
        "--budget",
        type=float,
        default=600.0,
        help="Per-trial training wall-clock budget in seconds.",
    )
    p.add_argument(
        "--search-budget",
        type=float,
        default=None,
        help="Global wall-clock budget for the entire study in seconds.",
    )
    p.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of concurrent trials (requires shared SQLite storage).",
    )
    p.add_argument(
        "--families",
        default=None,
        help=(
            "Comma-delimited list of model families to search. "
            "Defaults to the domain's ALL_FAMILIES."
        ),
    )
    p.add_argument(
        "--no-warm-start",
        action="store_true",
        help="Skip enqueuing the domain's WARM_START_CONFIGS.",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory for TSV, JSONL, and SQLite outputs.",
    )
    p.add_argument(
        "--phase",
        type=int,
        choices=[1, 2],
        default=None,
        help=(
            "Phase 1 = tune family/HPs with fixed features. "
            "Phase 2 = ablate features with fixed family/HPs. "
            "Omit for joint search."
        ),
    )
    p.add_argument(
        "--phase2-family",
        default=None,
        help="Phase 2 only: the winning model family from phase 1.",
    )
    p.add_argument(
        "--phase2-params",
        default=None,
        help="Phase 2 only: JSON string of fixed model params from phase 1.",
    )
    p.add_argument(
        "--extract-phase1",
        type=Path,
        default=None,
        help="Path to a phase 1 results DB; print best-trial params and exit.",
    )
    p.add_argument(
        "--format-warm-start",
        type=Path,
        default=None,
        help=(
            "Path to completed study DB; format best trial as a "
            "WARM_START_CONFIGS entry and exit."
        ),
    )
    return p
