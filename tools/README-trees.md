# Isolated experiment trees

Tools for running comparative autoresearch experiments without cross-tree bias.

## Why

The main repo accumulates knowledge across sessions (`learnings.md`, ~150 session tags, committed plans/results, persistent `<variant>/best` pointers). That's great for incremental progress but fatal for controlled comparisons: any agent starting on `main` inherits every prior session's conclusions via `learnings.md` alone.

These tools create per-tree isolated workspaces so that a fresh agent in tree A sees nothing from tree B's history.

## Concepts

- **Template**: this repo (`routee-autoresearch`). Holds the harness — fixed evaluation (`fixed_utils.py`), data, protocol (`program.md`), domain rules (`domain.md`), and the `train.py` scaffold.
- **Tree**: a separate git repo in `~/repos/routee-autoresearch-trees/<name>/`. Contains only the harness plus a single scaffold commit. No prior tags, no prior branches, no accumulated `learnings.md`. The agent works inside this tree.
- **Registry**: `~/repos/routee-autoresearch-trees/registry.jsonl`. One line per tree recording name, template sha, powertrain, and which seed files were used. Use it to compare trees later.

## Create a tree

```
tools/new_experiment_tree.sh \
    --name unguided-01 \
    --powertrain bev
```

Creates `~/repos/routee-autoresearch-trees/unguided-01/` with an empty `learnings.md`, an empty `seed.md`, the harness at the current template HEAD, and a `POWERTRAIN = "bev"` in `train.py`.

With intervention:

```
tools/new_experiment_tree.sh \
    --name guided-01 \
    --powertrain bev \
    --seed-md ./hints/guided-01-seed.md
```

With pre-loaded learnings (e.g., to simulate mid-experiment starts or test with partial priors):

```
tools/new_experiment_tree.sh \
    --name bootstrap-01 \
    --powertrain bev \
    --seed-learnings ./priors/bev-minimal.md
```

## Run an agent in the tree

```
cd ~/repos/routee-autoresearch-trees/unguided-01
# start your agent here — program.md inside the tree is the protocol it follows
```

The tree is a standalone git repo. The agent commits, tags, and branches inside it. Nothing leaks to the template repo or to sibling trees.

## Verify isolation

```
cd ~/repos/routee-autoresearch-trees/unguided-01
git log --all --oneline    # → exactly one commit (initial scaffold)
git tag -l                 # → empty
git branch -a              # → only main
cat learnings.md           # → empty (or the seed file's content)
cat .tree-meta.json        # → provenance
```

## Sync a harness change into a live tree

Rare — only when a harness bug fix or protocol change must reach an in-progress tree. Breaks strict reproducibility from the original scaffold, so use deliberately.

```
tools/sync_harness.sh --tree ~/repos/routee-autoresearch-trees/unguided-01
```

Copies harness files (`program.md`, `fixed_utils.py`, `pixi.*`, etc.) from the template's HEAD into the tree and commits as `harness: sync from template@<sha>`. Does not touch `train.py`, `learnings.md`, `seed.md`, `results/`, or `plans/`.

## What each tree contains

Copied from template (scaffold, identical across trees):

- `program.md`, `domain.md`, `fixed_utils.py`, `train.py` (scaffold)
- `pixi.toml`, `pixi.lock`, `Dockerfile`, `dprint.json`
- `templates/`, `example/`
- `.gitignore`, `.gitattributes`, `README.md`

Replaced per tree:

- `learnings.md` — empty by default, caller-overridable via `--seed-learnings`
- `seed.md` — empty by default, caller-overridable via `--seed-md`
- `train.py` — `POWERTRAIN` constant set per `--powertrain`
- `results/{bev,ice,phev}/.gitkeep` — fresh skeleton
- `.tree-meta.json` — tree provenance (template sha, powertrain, seed sources, creation time)

Not copied — symlinked:

- `data/` → template's `data/` (read-only per protocol; saves disk)

## Design notes

- **Why separate repos, not orphan branches?** `git log --all`, `git tag -l`, `git for-each-ref`, and reflog all happily cross orphan branches in a shared repo. Hiding refs is porous. Separate `.git` directories are the only airtight boundary git provides for free.

- **Why symlink `data/`?** The parquets are large and read-only per protocol (`fixed_utils.py` loads them, nothing writes them). Symlinking saves disk and avoids silent divergence. If you ever need fully archivable trees, replace the symlink with a copy.

- **Why not sync harness updates automatically?** Reproducibility. Each tree should be rebuildable from a single scaffold commit. Automatic sync would mean trees silently drift based on when you last touched them. The explicit `sync_harness.sh` exists for the rare case where a harness fix must propagate.
