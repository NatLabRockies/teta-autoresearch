# optimizers/

Optuna-backed metaheuristic search drivers. Each subpackage under
`optimizers/` is one sampler method with the same CLI surface.

## Running

```bash
pixi run python -m optimizers.tpe.search --tag <tag> --partition <p> --n-trials 200
pixi run python -m optimizers.cmaes.search --tag <tag> --partition <p> --n-trials 200
pixi run python -m optimizers.random.search --tag <tag> --partition <p> --n-trials 200
```

- `--domain` defaults to `routee`. Override to target another domain.
- `--partition` is required unless `--data-path` is passed explicitly.
- `--n-trials` caps total finished trials, including resumed trials from
  a previous run under the same `--tag`.
- `--budget <s>` is per-trial wall-clock; `--search-budget <s>` is a
  whole-study timeout. Either can be used alone or together.
- `--no-warm-start` skips enqueuing the domain's `WARM_START_CONFIGS`.
- `--families rf,xgb,cnn` restricts the search to those model families
  (validated against the domain's `ALL_FAMILIES`).

## Output

Each run writes to `--results-dir` (default `results/`):

```
search-<tag>-<method>.db       Optuna SQLite storage (study is named <tag>-<method>)
search-<tag>-<method>.tsv      quick-glance summary (same columns as program.md's TSV)
search-<tag>-<method>.jsonl    per-trial structured log (same format as program.md's JSONL)
timing-<tag>-<method>.log      per-trial start/end UTC timestamps
```

Naming includes the method so that running `tpe` and `cmaes` under the
same `--tag` inside one tree produces independent, comparable artifacts.

## Resuming

Run the same command again (same `--tag`, same `--results-dir`, same
method). Optuna detects the existing study in the SQLite file and resumes
from the next trial. Warm-start enqueue is skipped on resumed runs.

## Two-phase search

Some domains split search into a family/HP phase followed by a feature
ablation phase. See `--phase`, `--phase2-family`, `--phase2-params`, and
`--extract-phase1` in `optimizers/common/cli.py`.

## Adding a method

See `EXTENDING.md`. Add `optimizers/<method>/{__init__.py,sampler.py,search.py}`
and wire the name into `tools/new_experiment_tree.sh`'s `--optimizer`
regex.
