# autoresearch

A template for running **autonomous research sessions** that iteratively improve an ML model.
An agent (e.g. Claude Code) edits a single scaffold, `train.py`, one atomic change at a time —
proposing a hypothesis, training, evaluating against held-out data, keeping what helps and
reverting what doesn't — tagging every experiment and logging its reasoning as it goes.

The reference problem is RouteE vehicle energy prediction: predict per-link energy consumption
for use inside a shortest-path router. Adapting it to your own problem means editing three files
(see [Adapting it](#adapting-it)).

## Repo layout

```
program.md       the experiment protocol the agent follows
domain.md        the problem: context, inference-time constraints, guardrails
seed.md          the human's brief for a session (empty by default)
learnings.md     findings accumulated across sessions
train.py         the scaffold under optimization — the only file the agent edits
fixed_utils.py   the fixed point: train/test split and metrics, never edited
data/            dataset (not committed — see data/README.md)
plans/           one session plan per session, updated live as a progress log
results/         per-experiment TSV + JSONL, token usage
tools/           tree creation, token accounting
```

`fixed_utils.py` is the fixed point. The split and the metric definitions live there and are
off-limits to the agent, which is what makes results comparable across every experiment in a
tree.

## Quickstart

### 1. Create an isolated tree

Every run happens in a fresh git repo (a "tree") with exactly one commit, so the agent cannot see
prior sessions through `git log --all`, `git tag -l`, or an inherited `learnings.md`. That makes
each tree an independent sample of what the method finds rather than a continuation of the last
run.

```bash
tools/new_tree.sh ../bev-run-01
```

Options:

```bash
# hand the session a written brief
tools/new_tree.sh ../bev-run-02 --seed-md ~/notes/brief.md

# omit domain.md entirely — no domain context, no constraints
tools/new_tree.sh ../bev-run-03 --no-domain
```

The tree symlinks `data/` back to this template's copy, so historical experiment tags resolve
their data paths unchanged.

### 2. Run a session

```bash
cd ../bev-run-01
claude
# then: "Have a look at program.md and let's kick off a new experiment session"
```

The agent runs until a session limit is hit (50 experiments or 8 hours), then writes its findings
to `learnings.md` and records token usage.

### 3. Read the results

Every experiment is a commit and a tag, so the whole session is addressable after the fact:

```bash
git tag -l                          # every experiment
git show bev-may8/exp4              # the change itself
cat results/results-bev-may8.tsv    # the metrics, one row per experiment
cat results/experiments-bev-may8.jsonl   # hypothesis, observation, reasoning
```

## How scoring works

`fixed_utils.evaluate()` returns a dict of named metrics, and that dict is the single source of
truth for what "better" means. The protocol never hardcodes a metric name — the printed output,
the TSV columns, and the JSONL fields all follow from its keys.

Lower is better for every metric, and a change is kept only if it **Pareto-dominates** the current
best: every metric no worse, at least one strictly better. The reference domain scores two —
link-level RMSE and trip-level total-energy RMSE — because a model can improve per-link variance
while getting trip totals worse, and only tracking both catches it.

## Adapting it

Three files, in this order:

1. **`fixed_utils.py`** — define your metrics in `evaluate()`. Everything downstream follows.
2. **`domain.md`** — describe the problem and, importantly, the constraints: what the model will
   and won't have available in deployment, and what it must not do to the data. An agent
   optimizing a number will find every gap you leave here.
3. **`train.py`** — a working baseline for your problem. Keep it simple; it is a starting point
   the agent will replace, not a finished model.

`program.md` is domain-agnostic and should not need editing.

## Development

```bash
pixi run check    # mypy, ruff, dprint, unit tests
pixi run test     # unit tests only
pixi run fix      # apply formatting
```

# Acknowledgments

This software is built on the "autoresearch" software by github user karpathy available here [link](https://github.com/karpathy/autoresearch) and distributed under the MIT license.

# Metadata

NLR Software Record # SWR 26-089.
