# RouteE Autoresearch — Metaheuristic Baseline

Automated hyperparameter search over a prescribed set of ML/DL model families
for vehicle energy rate prediction.  This is the metaheuristic counterpart to
the LLM-driven experiment loop described in `../program.md`.

## Approach

[Optuna](https://optuna.org) (TPE / CMA-ES / Random sampler) searches over model families, 
hyperparameters, and feature subsets. Three search modes are supported:

### Joint Search (default)

Searches all three axes simultaneously:

| Axis | What is searched |
|---|---|
| **Model family** | RF, ExtraTrees, HGBR, XGBoost, LightGBM, MLP, 1-D CNN, GRU |
| **Feature subset** | Binary inclusion flags over all 18 engineered domain features |
| **Hyperparameters** | Per-family HP ranges; namespaced to avoid surrogate cross-talk |

### Two-Phase Search (recommended for large feature spaces)

Splits the search into two sequential phases for faster convergence:

- **Phase 1**: Fix all features, search only model family + hyperparameters. 
  TPE's surrogate converges faster with a constant feature surface.
  
- **Phase 2**: Fix the winning family + hyperparameters from Phase 1, search only feature subsets.
  Each trial is a direct A/B test of feature inclusion, enabling clean ablation.

Two-phase typically requires fewer total trials than joint search to find competitive solutions.

### Common settings

Every trial trains a model within a configurable time budget and evaluates on
a held-out test set using the same `fixed_utils.evaluate` RMSE metric as the
LLM session.  Results are persisted to an SQLite study (resumable) and logged
to TSV + JSONL files in the same format as `../results/`.

The first two trials are seeded from the best known configurations in
`../learnings.md` (CNN RMSE 0.006126, RF RMSE 0.006400) to give Optuna's
surrogate model a strong prior (joint search and Phase 1 only; Phase 2 skips warm-start).

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

### Joint Search (simpler, all-in-one)

```bash
# 100 trials, 2-minute budget per trial, TPE sampler
./run.sh --tag apr15 --n-trials 100

# Resume an interrupted run (same tag picks up from where it left off)
./run.sh --tag apr15 --n-trials 200

# CMA-ES sampler, 5-minute budget (better for neural families)
./run.sh --tag apr15 --n-trials 50 --sampler cmaes --budget 300

# a true baseline run (no surrogate model)
./run.sh --tag apr15-rand --n-trials 200 --sampler random --no-warm-start
```

### Two-Phase Search (recommended)

```bash
# Phase 1: 150 trials, find best family + HPs with all features
./run.sh --tag apr17-p1 --n-trials 150 --phase 1 --budget 120

# After Phase 1 completes, extract best trial params for Phase 2:
pixi run python search.py --extract-phase1 results/search-apr17-p1.db
# Output will show: --phase2-family rf --phase2-params '{"rf_n_estimators": 1200, ...}'

# Phase 2: 80 trials, ablate features with best family + HPs from Phase 1
./run.sh --tag apr17-p2 --n-trials 80 --phase 2 \
  --phase2-family rf \
  --phase2-params '{"rf_n_estimators": 1200, "rf_max_depth": 25, "rf_min_samples_split": 5, ...}'

# Results are in results/search-apr17-p2.tsv and results/search-apr17-p2.jsonl
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
| `--phase` | None | Search phase: `1` (tune family+HPs) or `2` (ablate features). Omit for joint search |
| `--phase2-family` | None | For phase 2: model family from Phase 1 (e.g. `rf`, `cnn`) |
| `--phase2-params` | None | For phase 2: JSON string of fixed HPs (e.g. `'{"rf_n_estimators": 1200, ...}'`) |
| `--extract-phase1` | None | Path to phase 1 SQLite DB; extract and print best trial params for phase 2, then exit |

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
| Quick smoke-test | `--n-trials 10 --budget 30` (joint) |
| Tabular-only exploration | `--n-trials 200 --budget 120` (joint) |
| Full search including NNs | `--n-trials 100 --budget 600` (joint) or `--phase 1 150 trials + phase 2 80 trials` |
| Parallel (4 workers) | 4× `./run.sh --tag X --n-trials 50 --parallel 1` in separate terminals |

**When to use two-phase search:**
- You have many feature candidates (18+ features) and TPE's surrogate is struggling to converge
- You want clear ablation studies: which features actually help?
- You prefer sequential execution: phase 1 finds the algorithm, phase 2 finds the features

**When to use joint search:**
- You want simplicity: one command, one study
- Your feature space is small (< 10 features) 
- You're doing random or CMA-ES sampling (less sensitive to dimensionality than TPE)

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
