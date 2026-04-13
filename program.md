# Autoresearch

This is an experiment to research better ML models. See `domain.md` for domain context and constraints.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `routee-autoresearch/<tag>` must not already exist — this is a fresh run.
1. **Create the branch**: `git checkout -b routee-autoresearch/<tag>` from current main.
1. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `learnings.md` - Accumulated findings from previous experiment sessions. Use this to avoid repeating dead ends and build on what works.
   - `seed.md` - Notes and ideas for this experiment session. Do not modify.
   - `domain.md` - Explanation of the domain context and constraints. Do not modify.
   - `fixed_utils.py` — fixed constants, data prep, evaluation. Do not modify.
   - `train.py` — A file you modify. Model architecture, optimizer, training.
1. **Initialize results files**: Create all of these with just their headers/empty:
   - `results/results-<tag>.tsv` — header row only
   - `results/experiments-<tag>.jsonl` — empty file
   - `results/exp-timing-<tag>.log` — empty file
1. **Create a session plan**: You MUST create `plans/plan-<tag>.md` using the template at `templates/plan-template.md`. Spend time researching the current state of knowledge, reviewing `learnings.md` and `seed.md`, then fill in session goals, planned experiments, and constraints. This file is updated throughout the session as a progress log.
1. **Run the baseline**: Run `train.py` as-is and record the baseline result. Then tag:
   ```
   git tag <tag>/baseline
   git tag <tag>/exp0
   git tag -f <tag>/best
   ```
1. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

The training script runs for a **fixed time budget of 10 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `pixi run python train.py`.

**What you CAN do:**

- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, features, etc.

**What you CANNOT do:**

- Modify `fixed_utils.py` or `domain.md`. It is read-only. It contains the fixed evaluation and data loading.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate` function in `fixed_utils.py` is the ground truth metric.

**The goal is simple: get the lowest rmse.** Since the time budget is fixed, you don't need to worry about training time — it's always 10 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 rmse improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 rmse improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**Atomic changes**: Make sure that every experiment only includes one atomic change that can be pointed to as a cause of the resulting model improvement. For example, if you decide to add a new feature, only add a single feature and see how the model reacts rather than adding two features since we won't know which feature resulted in the improvement.

## Output Format

Once the script finishes it prints a summary like this:

```
rmse: 0.039590
total_seconds: 9.4
features: speed_mph,grade_percent
```

To extract results from the log: `grep "^rmse:" run.log`

## Results TSV Format

The TSV has 4 columns:

```
commit	rmse	status	description
```

1. git commit hash (short, 7 chars)
2. rmse (e.g. 0.039590) — use 0.000000 for crashes
3. status: `keep`, `discard`, or `crash`
4. short text description of what this experiment tried

Example:

```
commit	rmse	status	description
a1b2c3d	0.039590	keep	baseline
b2c3d4e	0.035200	keep	increase LR to 0.04
c3d4e5f	0.041000	discard	switch to GeLU activation
d4e5f6g	0.000000	crash	double model width (OOM)
```

## Experiment Reasoning (JSONL)

For every experiment, append one JSON line to `results/experiments-<tag>.jsonl`. This is the structured reasoning record that captures _why_ experiments were tried and what was learned.

**Format** (one line per experiment, no pretty-printing):

```json
{
  "exp": 15,
  "commit": "5959293",
  "parent_best": "0271671",
  "status": "keep",
  "rmse": 0.130,
  "best_rmse_before": 0.013419,
  "delta_pct": -35.2,
  "description": "capture previous link speed",
  "hypothesis": "link sequencing captures dynamics that speed/grade averages miss",
  "observation": "1% RMSE reduction",
  "reasoning": "Using link sequencing captures dynamics about the vehicle that cannot be captured when veiwing the link independently",
  "tags": ["feature-engineering"]
}
```

**Fields:**

- `exp`: experiment number (int)
- `commit`: short hash of the experiment commit (string)
- `parent_best`: short hash of the current best commit before this experiment (string)
- `status`: `keep`, `discard`, or `crash` (string)
- `rmse`: observed RMSE (float, use 0.0 for crashes). For multi-vehicle, use an object: `{"bolt": 0.008, "camry": 0.015}`
- `best_rmse_before`: the best RMSE before this experiment (float or object)
- `delta_pct`: percent change from best (-5.0 means 5% improvement). Use `null` for crashes
- `description`: what was changed (same as TSV, 1 line)
- `hypothesis`: what you expect to happen and why — write BEFORE running (1-2 sentences)
- `observation`: what actually happened — write AFTER running (1-2 sentences)
- `reasoning`: why it worked/failed and what it teaches for future experiments (1-2 sentences)
- `tags`: category labels like `feature-engineering`, `hypertuning`, `architecture`, `data-filtering`, `breakthrough`, `dead-end`

Keep hypothesis/observation/reasoning concise — 1-2 sentences each. This is a decision log, not a paper.

## Logging results

When an experiment is done, log it to BOTH:

1. `results/results-<tag>.tsv` — the quick-glance summary (tab-separated, NOT comma-separated)
2. `results/experiments-<tag>.jsonl` — the structured reasoning record (one JSON line, appended)

The TSV has a header row with columns for: git commit hash (short, 7 chars), metric columns (`rmse`), status (`keep`, `discard`, or `crash`), and a short text description. Use 0.000000 for crashes.

## Git Tags

Every experiment session uses three tiers of git tags for history tracking and forking:

**Tier 1 — Every experiment (automatic):**

- `<tag>/expN` — lightweight tag on every experiment commit, created before running. These make every experiment addressable for forking.

**Tier 2 — State tracking (automatic):**

- `<tag>/baseline` — after the first baseline run, never moved
- `<tag>/best` — force-updated (`git tag -f`) to the latest best-performing commit after each improvement

**Tier 3 — Milestones (judgment call):**

- `<tag>/milestone-<desc>` — for breakthroughs (>10% improvement, new approach working, qualitative shift). Example: `apr2/milestone-link-sequence`

Push tags with: `git push --tags`

## The experiment loop

The experiment runs on a dedicated branch (e.g. `routee-autoresearch/mar5`).

LOOP FOREVER:

1. **Determine experiment number N** and review git state (current branch, current best commit)
2. **Form hypothesis**: Before editing code, decide what you're testing and why. Write this down mentally — it goes into the JSONL.
3. **Edit `train.py`** with one atomic experimental change
4. **Commit**: `git commit -m "expN: <short description>"`
5. **Tag**: `git tag <tag>/expN`
6. **Log start time**: `echo "expN start $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> results/exp-timing-<tag>.log`
7. **Run**: `pixi run python train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
8. **Log end time**: `echo "expN end $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> results/exp-timing-<tag>.log`
9. **Parse results**: `grep "^rmse:" run.log`. If empty, the run crashed — run `tail -n 50 run.log` to read the stack trace.
10. **Record results**: Append to both the TSV and the JSONL (with hypothesis, observation, reasoning)
11. **If RMSE improved** (lower):
    - Update best tag: `git tag -f <tag>/best`
    - Commit the updated results files: `git commit -m "log expN: <result summary>"`
    - Consider a milestone tag if this is a breakthrough (>10% improvement)
12. **If RMSE is worse, equal, or crashed:**
    - Restore train.py from best: `git checkout <tag>/best -- train.py`
    - Commit the revert + updated results files: `git commit -m "revert expN: <short reason>"`
13. **Update plan**: Add a progress line to `plans/plan-<tag>.md`
14. **Push**: `git push && git push --tags`
15. **Periodic learnings update** (~every 20 experiments): checkout main, update `learnings.md` with new findings, commit, push, checkout back to experiment branch

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working _indefinitely_ until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

## Forking

When you want to explore a divergent direction from a previous experiment without abandoning the current branch:

1. **Identify the fork point**: Find the tag `<tag>/expN` you want to fork from
2. **Create a new branch**: `git checkout -b routee-autoresearch/<tag>-fork-<desc> <tag>/expN`
3. **Initialize new results files**: Create fresh TSV, JSONL, and timing log for the fork
4. **Create a fork plan**: Create `plans/plan-<tag>-fork-<desc>.md` noting the fork point and rationale
5. **Continue the experiment loop** on the new branch

The original branch is untouched. The fork starts from the exact code state of expN.

**When to consider forking:**

- 5+ consecutive discards suggest the current direction is exhausted
- A discarded experiment showed promise in one metric but regressed another
- You want to try a fundamentally different approach (e.g. different model family) without losing current progress
- The human asks you to explore a specific past experiment further

## Cross-session learnings

`learnings.md` on the `main` branch accumulates insights across sessions. It is structured by vehicle, with sections for what works, what doesn't, best known configurations, and open questions.

**At session start:** Read `learnings.md` as part of the in-scope files. Use it to avoid repeating known dead ends and to build on proven approaches.

**During the session (~every 20 experiments):** Update `learnings.md` with new findings:

```
git stash
git checkout main
# update learnings.md
git add learnings.md
git commit -m "update learnings from <tag> session"
git push
git checkout routee-autoresearch/<tag>
git stash pop
```

**At session end:** Do a final learnings update before stopping.
