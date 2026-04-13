# Domain

**Context**
Our training data represents a single simulated vehicle, a Chevy Bolt, over drive cycle traces (typically 1hz). We take these point level results and aggregate them up to the trip/road segment level. Then, the road segments have attributes like total distance, average speed, average road gradiant, road classification, time to traverse.
The Chevy Bolt is a battery electric vehicle and as a result, the link energy can be negative (regenerative breaking).
The default feature set is speed and grade but you can experiment with any feature combination.

Note that when we're applying these models for inference, we often only have limited data (which is why we're developing these models in the first place).
Think about the inference environment as applying these models during a shortest path search in Google Maps where we only have limited information.
If you're considering any kind of link sequencing, we will only have the context of the previous links that have been traversed and know nothing about the future links that might be traversed.

## Output Format

Once the script finishes it prints a summary like this:

```
========================================
vehicle: 2017_Chevy_Bolt
========================================
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
