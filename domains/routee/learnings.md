# Learnings

Accumulated findings across experiment sessions. Read the **Cross-cutting insights** section at the start of every new session regardless of powertrain. Then read the section for the powertrain you're working on.

Each session targets exactly one powertrain (BEV, ICE, or PHEV). Branch and results files are namespaced by powertrain (see `program.md`). This file is the single shared memory across all powertrains — update the right subsection when logging new findings.

## Cross-cutting insights

These are findings that are about the training pipeline, optimization, or data representation itself — not about any specific vehicle. Treat them as defaults for any new powertrain session; only re-test if you have a concrete reason to suspect the answer is powertrain-dependent.

### Training pipeline / environment

- **`torch.compile` is incompatible with `ProcessPoolExecutor`** (pickle error). Don't use it in `train.py`.
- **`DataLoader(num_workers>0)` causes OneCycleLR step overflow** when running inside `ProcessPoolExecutor`. Keep `num_workers=0`.
- **`OneCycleLR` epochs must match the actual number of training epochs.** Setting `epochs=20` when only 15 fit in the budget keeps LR too high and hurts generalization.
- **Target normalization adds no benefit** with AdamW — the optimizer adapts to target scale.
- **Gradient clipping (`max_norm=1.0`) helps** at higher learning rates — prevents occasional gradient spikes.

### Loss / optimizer

- **MSE loss aligns with RMSE evaluation** — Huber is worse because it doesn't match the eval metric.
- **AdamW >> SGD+momentum** under this compute budget. SGD was 119% worse (too slow to converge in ~15 epochs). Adaptive LR is essential at this scale.
- **`weight_decay=1e-4`** is the sweet spot; `1e-3` underfits.

### Architecture principles (short training budget)

- **BatchNorm, LayerNorm, Dropout1d in conv, and input noise augmentation all hurt** when inputs are pre-normalized and only ~15 epochs fit in the budget. These are regularization techniques that need many epochs to pay back; the model is compute-limited, not overfitting.
- **SiLU/Swish massively worse than ReLU** for this CNN architecture — slower and destabilizes training.
- **Simpler heads beat deeper heads** under a fixed compute budget. Fewer params → more epochs in 10 minutes. E.g. `Linear(640, 128) → ReLU → Linear(128, 1)` beats adding a 256-dim layer.
- **Ensembles don't fit** under the 10-minute budget — splitting time between models leaves each undertrained. (Unlike RF where each tree is cheap; for NNs the per-model epoch count is the bottleneck.)
- **Global avg/max pooling and temporal attention pooling** are worse than simple flattening for very short sequences (≤ 5 steps) — flattening preserves positional info.

### Data representation

- **Predict `energy_rate_gge` (energy / miles), not `energy_gge` directly.** Division by miles amplifies errors on short links when the model predicts raw energy — rate is the better target. (This is math, not vehicle-specific.)
- **Pre-computed aggregations are redundant for CNNs and trees alike.** Rolling means, std, cumulative sums, deltas-of-deltas, absolute-values, and multiplicative interactions are all derivable from the raw lagged features by either model family. Only add derived features that encode information the model can't reach from raw columns (e.g. `sinuosity` requires parsing WKB geometry).
- **Geometry features worth extracting from WKB:** `sinuosity` (path length / straight-line) and `abs_bearing_delta` (turn angle magnitude from endpoints). Shape-detail features (`max_curvature`, `n_points`, raw bearing/lat/lon) don't help — most segments have only 2-4 geometry points, and tree splits on absolute geo don't generalize.
- **StandardScaler on sequence features beats QuantileTransformer** — QT distorts feature relationships the CNN uses.

### Open cross-cutting questions

- Does the CNN vs tree-model advantage hold for all powertrains, or is it BEV-specific? (Confirm early on the first ICE/PHEV session.)
- Does the 4-link speed-lookback saturation point generalize, or is it driver/regen-dynamics specific? (Worth retesting per powertrain since ICE has no regen.)
- Does `energy_rate_gge` behave well for PHEV when combining battery + fuel energy into one scalar, or does the mode-switching produce bimodal residuals that argue for a dual-target model?

## BEV (2017 Chevy Bolt)

Battery electric vehicle. Energy can be **negative** on a link (regenerative braking). Asymmetric, heavy-tailed target distribution.

### What works

- **Previous link speed features are the single biggest lever.** `prev_speed_mph` gave 24% improvement. Adding `speed_delta` (current - prev speed) gave another 7.4%. Deeper lookbacks (`prev2`, `prev3`, `prev4`) each added 2-5%. These capture acceleration/deceleration dynamics between links.
- **prev_miles** — previous link distance provides road-type transition context. 2.8% improvement.
- **grade_delta** — grade change between consecutive links captures elevation transition dynamics. 0.34% improvement. Tested removing it later — RMSE got worse, so it carries real signal.
- **time_seconds** — time to traverse the link. 0.1% improvement. Tested removing it — worse without it. Captures something beyond miles/speed ratio.
- **speed_accel (2nd derivative of speed)** — change in speed_delta between consecutive links. 0.91% improvement. Captures acceleration dynamics beyond what speed_delta alone provides.
- **prev_time_seconds** — previous link duration. 0.26% improvement. Complements prev_miles for road-type context.
- **abs_bearing_delta (turn angle magnitude)** — absolute direction change from previous link, computed from geometry endpoints. 0.73% improvement. Signed bearing_delta was also useful (2.05%) but abs form is simpler and captures the turn-severity signal.
- **prev_abs_bearing_delta** — prior turn angle. 0.22% improvement. Turn history carries signal even though deeper lookbacks (prev2) saturate.
- **sinuosity (road curvature ratio)** — path length / straight-line length from geometry. 0.28% improvement. Tested removing it later — RMSE got worse, still needed.
- **prev_sinuosity** — previous link's sinuosity. 0.09% improvement.
- **cum_backward_miles** — cumulative backward distance as a per-timestep feature (0 for current link, increasing for lookback). Gives CNN distance context for variable-length links. 0.42% improvement.
- **1D-CNN beats RF** — 3-layer Conv1d (128ch, k=3) over 5-link windows achieves RMSE 0.006152 vs RF 0.006400, a 3.9% improvement. CNN naturally captures sequential link patterns.
- **OneCycleLR with `max_lr=4e-3`** is the best LR schedule for the current best CNN config.
- **Batch size 2048** slightly better than 1024 (more epochs in time budget).
- **128 conv channels** is the capacity sweet spot for the 10-min budget (256 too slow: 9 epochs vs 13; 64 underfits).
- **3 conv layers** is the right depth (2 marginally worse, deeper not tested as beneficial).
- **RF: `max_samples=0.5`, `max_features=0.7`, `max_depth=None`, `n_estimators=2000`, `min_samples_split=10`.** Subsampling is critical regularization; unlimited depth with subsampling doesn't overfit.

### What doesn't work

- **prev_grade_percent / prev_grade_delta beyond 1 lookback** — grade history consistently adds no signal. Grade is geography-dependent, not driver-dependent. Tested repeatedly (exp3, exp19, exp30, apr13b/exp18, apr13b/exp36) — never helps.
- **prev5_speed_mph (5-link lookback)** — signal plateaus at 4 links. 5th lookback hurts due to noise and data loss from dropna on short journeys.
- **prev2_abs_bearing_delta** — bearing lookback saturates at 1 link (unlike speed, which benefits out to 4 links).
- **Static delta features (`speed_delta`, `grade_delta`, `speed_accel`)** — redundant for CNN when the raw sequence is available; the CNN can derive deltas from adjacent timesteps. Still useful for RF/MLP baselines.
- **speed_squared and interaction features (speed×grade, speed×sinuosity, speed×turn)** — RF and CNN both capture these natively.
- **Absolute-value features (grade_abs, speed_delta_abs)** — trees handle via split thresholds.
- **Pre-computed rolling means / std / cumulative sums** — both RF and CNN derive these from raw values.
- **miles_per_second** — redundant with speed_mph.
- **Raw bearing, lat, lon, max_curvature, total_curvature, n_points** — weak or no signal.
- **HistGradientBoosting and ExtraTrees** — both worse than RF for this target (noisy energy with heavy regen tails).
- **Transformer / GRU** — too slow under 10-min budget (4-7 epochs vs 13 for CNN).
- **Residual connections** — no help on a 3-layer CNN with seq_len=5 — too shallow to need them.
- **Multi-scale / inception CNN (k=1,3,5)** — no help; k=3 with 3 layers already covers most of a length-5 sequence.
- **Kernel size 5 first layer** — marginally slower, no RMSE win.
- **Linear head (no hidden layer)** — too simple; conv output needs at least one nonlinear combination.
- **Channel attention (squeeze-and-excitation)** — too few channels/timesteps for attention to help.
- **Higher LR (5e-3, 1e-2)** — diverges even with OneCycleLR. `4e-3` is the ceiling.
- **Batch size 4096** — reduced gradient noise hurts generalization at this scale.
- **Sequence length 7** — no better than 5 (consistent with 4-lookback saturation for speed).
- **Signed bearing_delta as a sequence feature** — adds noise; `abs_bearing_delta` already captures turn severity.
- **Predicting `energy_gge` instead of rate** — 1.55% worse. Division by miles amplifies errors on short links.
- **Virtual fixed-distance links (resampling lookback at fixed intervals, 10m or 80m per virtual link)** — 8-31% worse. Piecewise-constant interpolation creates redundant features and destroys real-link-boundary transition signals.
- **n_points (geometry vertex count)** — no signal for CNN or RF.
- **cum_backward_time** — redundant with cum_backward_miles (correlated through speed).
- **link_position** — domain constraint (not available at inference). Cost of removing it is only 0.16%, negligible.

### Best known configuration — CNN (apr16/exp18, RMSE 0.006099)

- 1D-CNN: 3 Conv1d layers (128 channels, kernel_size=3, padding=1), ReLU activations
- Head: `Linear(640, 128) → ReLU → Linear(128, 1)` — no dropout
- Sequence: 5-link window (current + 4 previous), 7 features per link
- Link features: `speed_mph, grade_percent, miles, time_seconds, sinuosity, abs_bearing_delta, cum_backward_miles`
- No static features (link_position removed per domain constraint)
- StandardScaler on sequence features
- AdamW, OneCycleLR `max_lr=4e-3`, `weight_decay=1e-4`, gradient clipping `max_norm=1.0`
- Batch size 2048, MSE loss
- ~16 epochs in 10-minute budget
- Pointer: `bev/best` tag

### Best known configuration — RF (apr13b/exp20, RMSE 0.006400)

- `RandomForestRegressor(n_estimators=2000, max_depth=None, min_samples_split=10, max_features=0.7, max_samples=0.5)`
- Features: `speed_mph, grade_percent, miles, prev_speed_mph, speed_delta, prev_miles, grade_delta, link_position, prev2_speed_mph, prev3_speed_mph, prev4_speed_mph, time_seconds, sinuosity, abs_bearing_delta, prev_sinuosity, prev_abs_bearing_delta, speed_accel, prev_time_seconds`
- Geometry features parsed in a single pass from WKB LineString (SRID 4326)
- Note: predates the domain constraint against `link_position`; a current-constraint RF baseline would drop it.

## ICE / Conventional (2016 Toyota Camry)

Internal-combustion vehicle. Energy is **always non-negative** (fuel consumption only, no regen). Expect a more symmetric, less heavy-tailed distribution than BEV. No sessions run yet.

### What works

- _(To be filled as sessions run. When starting, inherit the BEV CNN config as the baseline — cross-powertrain architectural lessons apply — then iterate on ICE-specific tuning.)_

### What doesn't work

- _(To be filled.)_

### Best known configuration

- _(None yet. `ice/best` tag will be created after the first ICE session.)_

### ICE-specific hypotheses worth testing early

- Since energy ≥ 0, a softplus / ReLU output activation or non-negativity constraint might help vs an unbounded linear output.
- Fuel consumption is dominated by engine efficiency curves (torque × RPM regimes) that correlate strongly with absolute speed and grade. The driver-dynamics signal (speed history) that dominates BEV may matter less here — grade and absolute speed may dominate. Worth explicitly retesting whether 4-link speed lookback still saturates at 4 or saturates earlier/later.
- Log-target (`log(1 + energy_rate_gge)`) might stabilize training given strictly non-negative, right-skewed distribution.

## PHEV

Plug-in hybrid. Has **both** a battery (`ess_kwh_out_ach`) and a fuel tank (`fs_kwh_out_ach`). Current preprocessing combines them into a single GGE target (sum of both). No raw data yet — scaffolding only.

### What works

- _(To be filled.)_

### What doesn't work

- _(To be filled.)_

### Best known configuration

- _(None yet. `phev/best` tag will be created after the first PHEV session.)_

### PHEV-specific hypotheses worth testing early

- **Combined-GGE vs dual-target.** Current default combines battery + fuel into one `energy_rate_gge`. PHEVs mode-switch between EV and hybrid operation, which may produce a bimodal residual distribution. Predicting two rates (battery GGE, fuel GGE) with a 2-output head is the most obvious first experiment after baseline.
- **Mode indicator.** If available, including an "EV-mode fraction on this link" feature (share of link traveled in pure EV) could resolve most of the bimodality without dual targets.
- Expect the BEV lessons about regen-aware features (speed_delta, speed_accel, negative energy) to partially apply during EV-mode portions; ICE lessons (grade dominance, engine-curve effects) to apply during hybrid-mode portions.

## Open questions (all powertrains)

- Would sub-link segmentation (splitting long links at geometry vertices) help?
- Is there value in treating different journey types differently (short urban vs long highway)?
- Would weather or time-of-day features help, if they could be joined in?
- Does a single joint model across all powertrains (with a powertrain-type embedding) beat three independent models? Worth exploring once each powertrain has its own `<powertrain>/best`.
