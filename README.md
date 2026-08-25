# autoresearch

A template for running **autonomous research sessions** that iteratively improve an ML model.
An agent (e.g. Claude Code) edits a single scaffold, `train.py`, one atomic change at a time —
proposing a hypothesis, training, evaluating against held-out data, keeping what helps and
reverting what doesn't — committing every experiment and logging its reasoning as it goes.

The reference problem is RouteE vehicle energy prediction: predict per-link energy consumption
for use inside a shortest-path router. Adapting it to your own problem means editing three files
(see [Adapting it](#adapting-it)).

## Repo layout

```
program.md       the experiment protocol the agent follows
domain.md        the problem: context, constraints, etc. 
learnings.md     findings accumulated across sessions
train.py         the scaffold under optimization — the only file the agent edits
harness.py       the fixed point: time budget, split, metrics, reporting — never edited
data/            dataset (not committed — see data/README.md)
plans/           one session plan per session, updated live as a progress log
results/         per-experiment TSV + JSONL, token usage, session transcripts
tools/           tree creation, token accounting, transcripts
```

Two Python files, and the split between them is the whole experimental control. `harness.py` is
the fixed point: the time budget, the train/test split, the metric definitions, and the report
format. 

`train.py` is everything else, and all of it is fair game. The agent can rewrite it entirely;
its only obligations are to score with `harness.evaluate()` and report with `harness.report()`.

## Quickstart

### 1. Create an isolated tree

Every run happens in a fresh git repo (a "tree") with exactly one commit, so the agent cannot see
prior sessions through `git log --all` or an inherited `learnings.md`. 

```bash
tools/new_tree.sh ../bev-run-01
```

The tree symlinks `data/` back to this template's copy, so a checkout of any historical
experiment commit resolves its data paths unchanged.
`TEMPLATE_COMMIT` records the full template commit hash used to create the tree.

### 2. Run a session

```bash
cd ../bev-run-01
claude
# then: "Have a look at program.md and let's kick off a new experiment session"
```

The agent runs until a session limit is hit (50 experiments or 8 hours), then writes its findings
to `learnings.md`, records token usage, and commits the session transcript.

### 3. Read the results

A session is a straight line of commits on `main`. The experiment number
and the commit hash are recorded with every result:

```bash
git log --oneline                          # every experiment, in order
git show <hash>                            # the change itself
cat results/results-bev-may8.tsv           # the metrics, one row per experiment
cat results/experiments-bev-may8.jsonl     # hypothesis, observation, reasoning
cat results/usage-by-exp-bev-may8.jsonl    # what each experiment cost, per model
cat results/transcript-audit-bev-may8.md   # how the session actually ran
```
## How scoring works

`harness.evaluate()` returns a dict of named metrics, and that dict is the single source of
truth for what the metrics are. 

## Token cost and transcripts

Two things are recorded about the session itself, both committed into the tree:

- `results/usage-<tag>.jsonl` — appended cumulative per-model token totals for the session (take
  the latest snapshot per model; do not sum). `results/usage-by-exp-<tag>.jsonl` — rewritten each
  run, one line per experiment per model, covering only that experiment's window, so these lines
  _do_ sum. Attribution is retrospective, from the `ended_at` stamps the agent already logs, so
  nothing has to happen during the experiment loop. Both files also carry
  `context_tokens_last` / `context_tokens_max` — how full the context window was when each
  experiment finished, and its peak during that experiment. These are occupancy readings rather
  than costs, so they are neither summed nor accumulated, and a sharp drop between consecutive
  experiments is auto-compaction resetting the window. Subagents are counted in the token totals
  but excluded from the context figures: each runs in a window of its own, so folding them in
  would report a peak that never happened.
- `results/transcript-<tag>/` — the raw Claude Code transcripts, verbatim, plus
  `results/transcript-audit-<tag>.md` summarizing tool calls, every operator prompt, and every
  path touched outside the tree.

Transcripts are stored through **git-LFS** (`.gitattributes` declares the filter;
`tools/new_tree.sh` runs `git lfs install --local` in every tree it creates.
Three consequences worth knowing up front:

- The checked-out file is the real transcript, but `git show <rev>:results/transcript-…` prints
  the LFS pointer instead. Use `git cat-file --filters <rev>:<path>` to read a historical version,
  or just read the working-tree copy.
- A tree with no remote keeps its LFS objects in its own `.git/lfs/objects` — fully local and
  fully committed. Pushing such a tree later needs an LFS-capable remote.
- A committed transcript is readable by a _later_ session in the same tree, the same way
  `learnings.md` already carries across sessions there.

## Adapting it

Three files, in this order:

1. **`harness.py`** — define your metrics in `evaluate()`, and adjust the split and the time
   budget if your problem needs different ones. Everything downstream follows from `evaluate()`'s
   keys. 
2. **`domain.md`** — describe the problem; the constraints (what the model will and won't have
   available in deployment, and what it must not do to the data);
3. **`train.py`** — a working baseline for your problem. Keep it simple; it is a starting point
   the agent will replace, not a finished model. Set `MODEL_FAMILY`, `LINK_FEATURES`, `TARGET`
   and `DATA_PATH`, load your data, fit something, and hand the results to `report()`. There is
   no interface to implement — the agent is free to restructure the whole file.

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
