# RouteE domain

The reference domain for this template: learning ML models that predict
per-link vehicle energy for use inside a shortest-path router.

## What the data is

Simulated 1 Hz drive-cycle traces aggregated to the road-segment level.
Each row is one link with features (average speed, grade, distance,
geometry) and a target energy per mile in gasoline-gallon-equivalent.

- `data/processed/2017_Chevy_Bolt.parquet` — BEV partition (regenerative
  braking → negative link energy, heavy-tailed distribution).
- `data/processed/2016_Toyota_Camry.parquet` — ICE partition (fuel
  consumption only, non-negative).
- PHEV is listed but not yet included.

## Partition axis

`powertrain`: one of `bev`, `ice`, `phev`. A session targets exactly one.
See `domain.md` for the semantic rules (inference-time features, what can
and cannot be used). `domain.json` encodes the same partitioning for the
tooling.

## What gets modified where

- `train.py` — LLM-editable scaffold. Trees start with a single partition
  selected by the `POWERTRAIN` constant.
- `learnings.md` — cross-session knowledge base. Never modified inside a
  tree (trees ship empty) — only on the template's `main` branch.
- `seed.md` — per-session hints. Not modified; replaced per tree.
- `search/` — RouteE-specific glue for optimizer mode: search space,
  feature engineering, trainer dispatch. Consumed by `optimizers/`.

## Running experiments

```bash
# LLM session (inside a tree)
cd ~/repos/routee-autoresearch-trees/<tree>
pixi run python train.py

# Optimizer session (inside an optimizer tree)
cd ~/repos/routee-autoresearch-trees/<tree>
pixi run python -m optimizers.tpe.search --tag <tag> --n-trials 200 --partition bev
```

See the top-level `README.md` for how to create trees.
