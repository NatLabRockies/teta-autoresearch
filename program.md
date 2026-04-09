# routee-autoresearch

This is an experiment to research better model architectures for RouteE Powertrain.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b routee-autoresearch/<tag>` from current main.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `prepare.py` — fixed constants, data prep, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training.
4. **Start the token usage collector**: Launch the OTLP collector in the background so it captures token usage for the session:
   ```bash
   pixi run python otel_collector.py --tag <tag> &
   ```
5. **Initialize results.tsv**: Create `results/results-<tag>.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `pixi run python train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, features, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation and data loading.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate` function in `prepare.py` is the ground truth metric.

**Context**
Our training data represents two simulated vehicles, a Chevy Bolt and a Toyota Camry over road segments. They road segments have attributes like total distance, average speed, average road gradiant, road classification, time to traverse.
These link records were derived from running the FastSIM simualtor over 1Hz drive cycles that had been map matched to a real road network and then aggregating the attributes.
Right now both vehicles use the same model architecture but it's okay to use different architectures for each vehicle. 
Notably each vehicle uses a different powertrain technology. The Chevy Bolt is a battery electric vehicle and as a result, the link energy can be negative (regenerative breaking).
The Toyota Camry is a conventional vehicle and therefore the link energy will never be negative.
The default feature set is speed and grade but you can experiment with any feature combination.

Note that when we're applying these models for inference, we often only have limited data (which is why we're developing these models in the first place).
If you're considering any kind of link sequencing, we will only have the context of the previous links that have been traversed and know nothing about the future links that might be traversed.

**External Research**
You can search the web for context related to this problem and try anything that seems promising

**The goal is simple: get the lowest rmse.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 rmse improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 rmse improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
========================================
vehicle: 2017_Chevy_Bolt
========================================
rmse: 0.039590
total_seconds: 9.4
features: speed_mph,grade_percent

========================================
vehicle: 2016_Toyota_Camry
========================================
rmse: 0.025323
total_seconds: 9.4
features: speed_mph,grade_percent
```

## Logging results

When an experiment is done, log it to `results/results-<tag>.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	camry_rmse bolt_rmse    status	description
```

1. git commit hash (short, 7 chars)
2. rmse achieved (e.g. 1.234567) — use 0.000000 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	camry_rmse	bolt_rmse   status	description
a1b2c3d	0.997900 	0.997900    keep	   baseline
b2c3d4e	0.993200 	0.993200    keep	   increase LR to 0.04
c3d4e5f	1.005000 	1.005000    discard	switch to GeLU activation
d4e5f6g	0.000000 	0.000000    crash	   double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `routee-autoresearch/mar5` or `routee-autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `pixi run python train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^rmse:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: create a new sub folder in the results directory and commit to git)
8. If rmse improved (lower), you "advance" the branch, keeping the git commit
9. If rmse is equal or worse, you git reset back to where you started
10. Push any commits to github

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
