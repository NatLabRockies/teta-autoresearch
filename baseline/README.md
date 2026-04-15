# RouteE Autoresearch — Metaheuristic Baseline

Automated hyperparameter search over a prescribed set of ML/DL model families
for vehicle energy rate prediction.  This is the metaheuristic counterpart to
the LLM-driven experiment loop described in `../program.md`.

## Approach

[Optuna](https://optuna.org) (TPE / CMA-ES / Random sampler) drives a joint
search over three orthogonal axes:

| Axis | What is searched |
|---|---|
| **Model family** | RF, ExtraTrees, HGBR, XGBoost, LightGBM, MLP, 1-D CNN, GRU |
| **Feature subset** | Binary inclusion flags over all 18 engineered domain features |
| **Hyperparameters** | Per-family HP ranges; namespaced to avoid surrogate cross-talk |

Every trial trains a model within a configurable time budget and evaluates on
a held-out test set using the same `fixed_utils.evaluate` RMSE metric as the
LLM session.  Results are persisted to an SQLite study (resumable) and logged
to TSV + JSONL files in the same format as `../results/`.

The first two trials are seeded from the best known configurations in
`../learnings.md` (CNN RMSE 0.006126, RF RMSE 0.006400) to give Optuna's
surrogate model a strong prior.

## Prerequisites

[pixi](https://prefix.dev/docs/pixi/overview) must be installed.  The
`baseline/` directory has its own `pixi.toml` that adds Optuna, XGBoost, and
LightGBM on top of the parent workspace's packages.

```bash
# Install the baseline environment (one-time)
cd baseline/
pixi install
```

## Quick start

```bash
# 100 trials, 2-minute budget per trial, TPE sampler
./run.sh --tag apr15 --n-trials 100

# Resume an interrupted run (same tag picks up from where it left off)
./run.sh --tag apr15 --n-trials 200

# CMA-ES sampler, 5-minute budget (better for neural families)
./run.sh --tag apr15 --n-trials 50 --sampler cmaes --budget 300

# Random baseline (no surrogate model)
./run.sh --tag apr15-rand --n-trials 200 --sampler random --no-warm-start
```

## Arguments

| Flag | Default | Description |
|---|---|---|
| `--tag` | *(required)* | Run tag; names the study, TSV, JSONL, and SQLite DB |
| `--n-trials` | 100 | Total trials to run (resumed runs count existing trials) |
| `--sampler` | `tpe` | `tpe`, `cmaes`, or `random` |
| `--budget` | 120 | Per-trial training time in seconds |
| `--parallel` | 1 | Concurrent trials (share same SQLite DB across processes) |
| `--data-path` | `../data/processed/2017_Chevy_Bolt.parquet` | Parquet input |
| `--results-dir` | `results/` | Output directory |
| `--no-warm-start` | false | Skip seeding known-good configs as trial 0 and 1 |

## Output files

All outputs land in `results/` (created automatically):

```
results/
  search-<tag>.db      SQLite Optuna study — resume with same --tag
  search-<tag>.tsv     trial, rmse, status, description (tab-separated)
  search-<tag>.jsonl   per-trial structured log (one JSON line each)
```

The TSV and JSONL formats match the LLM session logs in `../results/` for
direct comparison.

## File structure

```
baseline/
  pixi.toml            Python environment (optuna, xgboost, lightgbm + torch + sklearn)
  run.sh               Convenience wrapper (forwards args to search.py)
  feature_pipeline.py  Data loading, feature engineering, sequence windowing
  search_space.py      Search space definition + warm-start seed configs
  models.py            Model training for all families (tabular + CNN/GRU)
  objective.py         Optuna objective factory + TSV/JSONL logging
  search.py            CLI entry-point
  results/             Created at runtime
```

## Budget guidance

| Goal | Suggested settings |
|---|---|
| Quick smoke-test | `--n-trials 10 --budget 30` |
| Tabular-only exploration | `--n-trials 200 --budget 120` |
| Full search including NNs | `--n-trials 100 --budget 600` |
| Parallel (4 workers) | 4× `./run.sh --tag X --n-trials 50 --parallel 1` in separate terminals |

For neural families (CNN, GRU) a budget of ≥300 s is recommended to get
enough training epochs.  With the default 120 s budget, tabular models will
dominate (they finish in seconds and get a proper evaluation), while neural
models will be undertrained — use `--budget 600` or longer to level the
playing field.

## Comparing with LLM sessions

The best known RMSE from `../learnings.md` at session start:

| Config | RMSE |
|---|---|
| 1-D CNN (apr14/exp22) | 0.006126 |
| RandomForest (apr13b/exp20) | 0.006400 |

Use these as the reference baseline when evaluating whether the metaheuristic
search finds better configurations.
