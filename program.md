# Autoresearch

This is an experiment to research better ML models. One session runs many experiments against a
single scaffold, `train.py`, keeping the changes that help and reverting the ones that don't.

Throughout this document `<tag>` is the session tag agreed at setup (e.g. `bev-may8`).

If `domain.md` is present, it defines the problem, the inference-time constraints, and what
counts as a legitimate change — read it first and treat it as binding. If `seed.md` is present,
it carries the human's brief for this session. Either file may be absent; the protocol below is
unchanged if so.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose something short and dated, e.g. `bev-may8`. The branch
   `autoresearch/<tag>` must not already exist — this is a fresh run.
1. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
1. **Read the in-scope files**: The repo is small. Read these for full context:
   - `domain.md` — problem definition and constraints. Do not modify. (May be absent.)
   - `seed.md` — the human's notes and ideas for this session. Do not modify. (May be absent.)
   - `learnings.md` — accumulated findings from previous sessions. Use it to avoid repeating
     dead ends and to build on what already works.
   - `fixed_utils.py` — the data split and the evaluation metrics. Do not modify.
   - `train.py` — the only file you edit.
1. **Seed `train.py` from the best known configuration**: if the persistent `best` tag exists,
   `git checkout best -- train.py`. Otherwise use `main`'s `train.py` as-is.
1. **Initialize results files**, with just their headers or empty:
   - `results/results-<tag>.tsv` — header row only
   - `results/experiments-<tag>.jsonl` — empty file
1. **Create a session plan**: You MUST create `plans/plan-<tag>.md` using the template at
   `templates/plan-template.md`. Spend time reviewing `learnings.md` and `seed.md` first, then
   fill in session goals, planned experiments, and constraints. This file is updated throughout
   the session as a progress log.
1. **Run the baseline**: Run `train.py` as-is and record the result. Then tag:
   ```
   git tag <tag>/baseline
   git tag <tag>/exp0
   git tag -f <tag>/best
   ```
1. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Metrics: the contract

`fixed_utils.evaluate()` returns a dict of named metrics. **That dict is the single source of
truth for what "better" means** — this protocol never hardcodes a metric name, and neither
should you. Everything downstream follows from its keys:

- `train.py` prints one `name: value` line per key, plus a single machine-readable
  `metrics: {...}` JSON line that is the parse target.
- The results TSV has one column per key, in that order.
- The JSONL records each metric's value and its percent change.

**Lower is better for every metric.** A change is **`keep`** only if it **Pareto-dominates** the
current best: _every_ metric is no worse, and _at least one_ is strictly better. If any metric
regresses, the change is **`discard`**, even if another improved. If all metrics tie exactly,
discard unless the change is a simplification (see below).

Do not privilege one metric over another. When metrics disagree, that disagreement is a finding
— record which one moved and why you think they split, because it usually points at the next
experiment.

## Experimentation

The training script runs for a **fixed time budget of 10 minutes** (wall clock training time,
excluding startup/compilation). You launch it as: `pixi run python train.py`.

**What you CAN do:**

- Modify `train.py` — this is the only file you edit. Model architecture, optimizer,
  hyperparameters, features: all fair game.

**What you CANNOT do:**

- Modify `fixed_utils.py`, `domain.md`, or `seed.md`. They are read-only. `fixed_utils.py`
  contains the fixed data split and the ground-truth metrics.
- Change how the model is scored, or which rows it is scored on. Filtering, reweighting, or
  dropping data before the split changes the exam rather than the model, and any improvement it
  produces is not real.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds
ugly complexity is not worth it. Conversely, removing something and getting equal or better
results is a great outcome — that's a simplification win. When evaluating whether to keep a
change, weigh the complexity cost against the improvement magnitude. A tiny improvement that
adds 20 lines of hacky code? Probably not worth it. The same improvement from deleting code?
Definitely keep. Metrics tied but the code is much simpler? Keep.

**Atomic changes**: Make sure that every experiment only includes one atomic change that can be
pointed to as a cause of the resulting model improvement. For example, if you decide to add a
new feature, only add a single feature and see how the model reacts rather than adding two
features since we won't know which feature resulted in the improvement.

**Stay in budget**: a run that exceeds the time budget is a `crash`, not a slow `keep`. Do not
keep a change that only fits by going over.

## Output Format

Once the script finishes it prints a summary like this:

```
rmse: 0.039590
trip_rmse: 0.421000
metrics: {"rmse": 0.03959, "trip_rmse": 0.421}
total_seconds: 9.4
features: speed_mph,grade_percent
```

The per-metric lines are for reading; the `metrics:` line is the one to parse, and it contains
exactly what `evaluate()` returned. `total_seconds` and `features` are metadata, not metrics.

To extract results from the log: `grep "^metrics: " run.log`

## Results TSV Format

Tab-separated, NOT comma-separated. Columns are the commit, then one column per metric in
`evaluate()` order, then status and description:

```
commit	rmse	trip_rmse	status	description
a1b2c3d	0.039590	0.421000	keep	baseline
b2c3d4e	0.035200	0.395000	keep	increase LR to 0.04 (both improved)
c3d4e5f	0.034800	0.402000	discard	GeLU activation (rmse better, trip_rmse worse)
d4e5f6g	0.000000	0.000000	crash	double model width (OOM)
```

1. git commit hash (short, 7 chars)
2. one column per metric — use 0.000000 for crashes
3. status: `keep`, `discard`, or `crash`
4. short text description of what this experiment tried

## Experiment Reasoning (JSONL)

For every experiment, append one JSON line to `results/experiments-<tag>.jsonl`. This is the
structured reasoning record that captures _why_ experiments were tried and what was learned.

**Format** (one line per experiment, no pretty-printing):

```json
{
  "exp": 15,
  "commit": "5959293",
  "parent_best": "0271671",
  "status": "keep",
  "started_at": "2026-05-08T19:16:32Z",
  "ended_at": "2026-05-08T19:24:59Z",
  "metrics": { "rmse": 0.0130, "trip_rmse": 0.395 },
  "best_before": { "rmse": 0.013419, "trip_rmse": 0.421 },
  "delta_pct": { "rmse": -3.1, "trip_rmse": -6.2 },
  "description": "widen the hidden layer 32 -> 64",
  "hypothesis": "the model is under-capacity; the training loss is still falling when the budget ends",
  "observation": "both metrics improved, and trip_rmse moved considerably further than rmse",
  "reasoning": "extra capacity helped, and the lopsided movement suggests the two metrics are limited by different things — worth probing next",
  "tags": ["architecture"]
}
```

**Fields:**

- `exp`: experiment number (int)
- `commit`: short hash of the experiment commit (string)
- `parent_best`: short hash of the current best commit before this experiment (string)
- `status`: `keep`, `discard`, or `crash` (string)
- `started_at` / `ended_at`: UTC timestamps, `date -u +%Y-%m-%dT%H:%M:%SZ`
- `metrics`: every metric `evaluate()` returned (use `0.0` for crashes)
- `best_before`: the same keys, at the current best before this experiment
- `delta_pct`: per-metric percent change from best (`-5.0` means 5% improvement). `null` for
  crashes
- `description`: what was changed (same as TSV, 1 line)
- `hypothesis`: what you expect to happen and why — write BEFORE running (1-2 sentences)
- `observation`: what actually happened, across ALL metrics — write AFTER running (1-2 sentences)
- `reasoning`: why it worked/failed and what it teaches for future experiments (1-2 sentences).
  If metrics disagreed, say which dynamic you think caused the split
- `tags`: category labels like `feature-engineering`, `hypertuning`, `architecture`,
  `data-representation`, `breakthrough`, `dead-end`

Keep hypothesis/observation/reasoning concise — 1-2 sentences each. This is a decision log, not a
paper.

## Git Tags

**Per experiment (automatic):**

- `<tag>/expN` — lightweight tag on every experiment commit, created before running. These make
  every experiment addressable after the fact.

**Session state (automatic):**

- `<tag>/baseline` — after the first baseline run, never moved
- `<tag>/best` — force-updated (`git tag -f`) to the session's best commit after each improvement

**Across sessions (at session end):**

- `best` — force-updated to this session's final best commit if it beats the previous `best`.
  This is the pointer the next session seeds `train.py` from.

## The experiment loop

The experiment runs on a dedicated branch `autoresearch/<tag>`.

**Session limits**: A session ends when either **50 experiments** have been run or **8 hours of
wall-clock time** have elapsed since the session started (setup time excluded), whichever comes
first. When the limit is reached, do the final learnings update, update `best` if appropriate,
record token usage, and stop.

Loop until the session limit is reached:

1. **Determine experiment number N** and review git state (current branch, current best commit)
2. **Form hypothesis**: Before editing code, decide what you're testing and why. It goes into the
   JSONL, and writing it first is what makes the result interpretable.
3. **Edit `train.py`** with one atomic experimental change
4. **Commit**: `git commit -m "expN: <short description>"`
5. **Tag**: `git tag <tag>/expN`
6. **Record the start time**: `date -u +%Y-%m-%dT%H:%M:%SZ`
7. **Run**: `pixi run python train.py > run.log 2>&1` (redirect everything — do NOT use tee or
   let output flood your context)
8. **Record the end time**
9. **Parse results**: `grep "^metrics: " run.log`. If that line is missing, the run crashed —
   `tail -n 50 run.log` to read the stack trace.
10. **Record results**: Append to both the TSV and the JSONL (with hypothesis, observation,
    reasoning, and the timestamps from steps 6 and 8)
11. **If the result Pareto-dominates** (every metric no worse, at least one strictly better):
    - Update best tag: `git tag -f <tag>/best`
    - Commit the updated results files: `git commit -m "log expN: <result summary>"`
12. **If any metric regressed, all tied, or the run crashed:**
    - Restore train.py from best: `git checkout <tag>/best -- train.py`
    - Commit the revert + updated results files: `git commit -m "revert expN: <short reason>"`
    - Note in the JSONL `observation` which metric regressed and by how much — that's the signal
      for the next hypothesis
13. **Update plan**: Add a progress line to `plans/plan-<tag>.md`
14. **Push**: `git push && git push --tags`
15. **Periodic learnings update** (~every 20 experiments): checkout main, update `learnings.md`
    with new findings, commit, push, checkout back to the experiment branch

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: if it's something dumb
and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is
fundamentally broken, skip it, log `crash` as the status, and move on.

**Keep going autonomously**: Do not pause to ask the human whether to continue. If you run out of
ideas before the session limit, think harder — re-read `learnings.md` and `seed.md` for new
angles, try combining previous near-misses, or try more radical architectural changes.

## Cross-session learnings

`learnings.md` on the `main` branch accumulates insights across sessions: what works, what
doesn't, the best known configuration, and open hypotheses.

**At session start:** Read it. Use it to avoid repeating known dead ends and to build on proven
approaches.

**During the session (~every 20 experiments):** Update it with new findings:

```
git stash
git checkout main
# update learnings.md
git add learnings.md
git commit -m "update learnings from <tag> session"
git push
git checkout autoresearch/<tag>
git stash pop
```

**At session end:** Do a final learnings update before stopping. Update the persistent `best` tag
to this session's final best commit if it beats the prior `best` (or create it if this is the
first session), and push tags.

Then record the session's LLM token usage:

```
pixi run python tools/token_usage.py --tag <tag>
```

This appends a cumulative per-model snapshot to `results/usage-<tag>.jsonl`. Each line is a
cumulative total at the time of the snapshot, not a delta — do **not** sum lines. Safe to run
mid-session too; every invocation re-reads transcripts from disk, so repeats cannot double-count.
