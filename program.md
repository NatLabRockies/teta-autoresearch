# Autoresearch 

This is an experiment to research better model architectures. See `domain.md` for domain context and constraints.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
1. **Create the branch**: `git checkout -b routee-autoresearch/<tag>` from current main.
1. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `domain.md` - Explanation of the domain context and constraints. Do not modify. 
   - `fixed_utils.py` — fixed constants, data prep, evaluation. Do not modify.
   - `adjustable_utils.py` — A file you can modify. Feature aggregation and engineering.
   - `train.py` — A file you modify. Model architecture, optimizer, training.
1. **Initialize results.tsv**: Create `results/results-<tag>.tsv` with just the header row. The baseline will be recorded after the first run.
1. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `pixi run python train.py`.

**What you CAN do:**

- Modify `train.py` or `adjustable_utils.py` — these are the only files you edit. Everything is fair game: model architecture, optimizer, hyperparameters, features, etc.

- You can search the web for domain specific ideas and research to inspire changes to the model architecture or the feature engineering.

**What you CANNOT do:**

- Modify `fixed_utils.py`. It is read-only. It contains the fixed evaluation and data loading.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate` function in `fixed_utils.py` is the ground truth metric.

**The goal is simple: get the lowest rmse.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 rmse improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 rmse improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary. See `domain.md` for the expected output format and example.

## Logging results

When an experiment is done, log it to `results/results-<tag>.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row with columns for: git commit hash (short, 7 chars), metric columns (see `domain.md` for specifics), status (`keep`, `discard`, or `crash`), and a short text description. Use 0.000000 for crashes.

See `domain.md` for the exact column definitions and an example.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `routee-autoresearch/mar5` or `routee-autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `pixi run python train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: use the grep pattern from `domain.md` to extract metrics from `run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: create a new sub folder in the results directory and commit to git)
8. If rmse improved (lower), you "advance" the branch, keeping the git commit
9. If rmse is equal or worse, you git reset back to where you started
10. Push any commits to github

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working _indefinitely_ until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

