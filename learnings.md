# Learnings

Accumulated findings across experiment sessions. Read this at the start of every new session to avoid repeating dead ends and to build on what works.

## 2017 Chevy Bolt (BEV)

### What works

- **Previous link speed features are the single biggest lever.** prev_speed_mph gave 24% improvement. Adding speed_delta (current - prev speed) gave another 7.4%. Deeper lookbacks (prev2, prev3, prev4) each added 2-5%. These capture acceleration/deceleration dynamics between links.
- **prev_miles** — previous link distance provides road-type transition context. 2.8% improvement.
- **grade_delta** — grade change between consecutive links captures elevation transition dynamics. 0.34% improvement. Tested removing it later — RMSE got worse, so it carries real signal.
- **link_position** — cumulative link count within a journey captures warm-up/cold-start effects. 0.17% improvement. Tested removing it — worse without it.
- **time_seconds** — time to traverse the link. 0.1% improvement. Tested removing it — worse without it. Captures something beyond miles/speed ratio.
- **speed_accel (2nd derivative of speed)** — change in speed_delta between consecutive links. 0.91% improvement. Captures acceleration dynamics beyond what speed_delta alone provides.
- **prev_time_seconds** — previous link duration. 0.26% improvement. Complements prev_miles for road-type context.
- **abs_bearing_delta (turn angle magnitude)** — absolute direction change from previous link, computed from geometry endpoints. 0.73% improvement. Signed bearing_delta was also useful (2.05%) but abs form is simpler and captures the turn-severity signal.
- **prev_abs_bearing_delta** — prior turn angle. 0.22% improvement. Turn history carries signal even though deeper lookbacks (prev2) saturate.
- **sinuosity (road curvature ratio)** — path length / straight-line length from geometry. 0.28% improvement. Tested removing it later — RMSE got worse, still needed.
- **prev_sinuosity** — previous link's sinuosity. 0.09% improvement.
- **max_samples=0.5** — subsampling acts as regularization and speeds up training. 0.41% improvement over no subsampling. Confirmed as sweet spot: 0.4 and 0.6 both worse.
- **max_features=0.7** — feature subsampling per split increases tree diversity. 0.38% improvement. Confirmed: 0.6 no better.
- **max_depth=None (unlimited)** — with subsampling regularization, unlimited depth doesn't overfit and captures deeper interactions.
- **n_estimators=2000** — diminishing returns but marginal gains. Training time ~511s, close to budget.

### What doesn't work

- **speed_squared** — RF already learns nonlinear speed relationships via tree splits. Polynomial features are redundant for tree models.
- **Interaction features (speed×grade, speed×sinuosity, speed×turn)** — all redundant. RF captures multiplicative interactions natively through tree splits.
- **prev_grade_percent / prev_grade_delta** — grade history consistently adds no signal. Grade is geography-dependent, not driver-dependent. Tested repeatedly across sessions (exp3, exp19, exp30, apr13b/exp18, apr13b/exp36) — never helps.
- **prev5_speed_mph (5-link lookback)** — signal plateaus at 4 links. 5th lookback hurts due to noise and data loss from dropna on short journeys.
- **prev2_abs_bearing_delta** — bearing lookback saturates at 1 link, unlike speed which benefits out to 4 links.
- **Absolute-value features (grade_abs, speed_delta_abs)** — RF handles absolute-value splits natively via split thresholds. Redundant.
- **Delta-of-delta features (sinuosity_delta, miles_delta, time_delta, prev_speed_delta)** — RF can derive these from raw lagged values. Redundant.
- **Pre-computed aggregations (speed_rolling_mean4, speed_std4, cumulative_miles, cumulative_time)** — RF can approximate these from individual lagged values. Redundant.
- **miles_per_second** — redundant with speed_mph (same information, different units).
- **Raw bearing, latitude, longitude** — weak signal; tree splits on absolute geo position don't generalize across 83%-unique road IDs.
- **max_curvature, total_curvature, n_geom_points** — geometry shape features beyond sinuosity + bearing_delta add no new signal. Most roads have only 2-4 geometry points anyway.
- **HistGradientBoostingRegressor** — 1.3% worse than RF. RF's bagging approach handles the noisy energy data better than sequential boosting.
- **ExtraTreesRegressor** — worse than RF by ~0.9%. The random-split variance hurts on this noisy target.
- **min_samples_split=5** — no improvement over 10, just slower training. Confirmed across sessions.
- **min_samples_leaf=5** — no improvement, slower.
- **max_samples=0.3, 0.4** — too aggressive, loses signal per tree. 0.5 is the sweet spot.
- **max_samples=0.6** — no improvement over 0.5 and slower.
- **n_estimators=400, 1500** — negligible gains beyond 2000's point on the curve relative to training-time cost.
- **Predicting energy_gge instead of rate** — 1.55% worse. Division by miles amplifies errors on short links. Direct rate prediction is better.

### Neural network findings (apr14 session)

- **1D-CNN beats RF.** A 3-layer Conv1d (128 channels, kernel_size=3) over 5-link journey windows achieves RMSE 0.006152 vs RF's 0.006400 — a 3.9% improvement. CNN naturally captures sequential patterns between consecutive links.
- **Per-link sequence features**: speed_mph, grade_percent, miles, time_seconds, sinuosity, abs_bearing_delta — these 6 features per timestep across a 5-link window are the CNN inputs. Link_position is a static feature concatenated after conv.
- **OneCycleLR with max_lr=3e-3** is the best LR schedule. CosineAnnealing with lr=3e-4 was 0.5% worse. Higher LR (1e-2) diverges.
- **Batch size 2048** slightly better than 1024 (more epochs in time budget).
- **Transformer too slow** — only 4 epochs vs CNN's 12-13. GRU also slow (7 epochs). CNN is the fastest architecture for fixed-length short sequences.
- **Ensembles don't work** under time budget — splitting time between models means each is undertrained. Unlike RF where each tree is cheap, NNs need many epochs.
- **BatchNorm hurts** with pre-normalized inputs and few epochs.
- **Residual connections no help** on 3-layer CNN with seq_len=5 — too shallow to need them.
- **MLP on flat features (same as RF)** gives RMSE 0.006343 — beats RF (0.006400) but worse than CNN (0.006152). Confirms CNN sequence modeling adds value beyond just NN optimization.
- **Static delta features (speed_delta, grade_delta, speed_accel)** redundant when CNN already has the raw sequence — CNN can derive deltas from adjacent timesteps.
- **Huber loss worse than MSE** — MSE training aligns better with RMSE evaluation metric.
- **Weight decay 1e-4** is sweet spot; 1e-3 underfits.
- **256 conv channels** too slow (9 epochs vs 13); 128 is the capacity sweet spot for 10-min budget.
- **Gradient clipping (max_norm=1.0)** helps with high LR — prevents gradient spikes during training. 0.42% improvement.
- **SiLU/Swish activation** massively worse than ReLU — slower and destabilizes training for this architecture.
- **QuantileTransformer** worse than StandardScaler — distorts feature relationships CNN was using.
- **Sequence length 7** no better than 5 — consistent with RF finding that speed lookback saturates at 4.
- **Multi-scale/inception CNN (k=1,3,5)** no help — with seq_len=5, kernel=3 already covers most of sequence.
- **Batch 4096** too large — reduced gradient noise hurts generalization. 2048 sweet spot.
- **2 conv layers** nearly tied with 3 layers but marginally worse — 3 layers is the right depth.
- **Global avg/max pooling** worse than flattening — flattening preserves positional info.
- **Temporal attention pooling** also worse than flattening — with only 5 timesteps, no need for learned attention.
- **LR 5e-3** too aggressive even with OneCycleLR. 3e-3 confirmed as sweet spot.
- **Signed bearing_delta** as sequence feature adds noise; abs_bearing_delta already captures turn severity.

### Neural network findings (apr16 session)

- **cum_backward_miles** — cumulative backward distance as a per-timestep feature (0 for current link, increasing for lookback). Gives CNN distance context for variable-length links. 0.42% improvement.
- **link_position removal** — domain constraint (not available at inference). Only 0.16% cost, negligible.
- **Removing head dropout** improves RMSE by 0.20% — with only 15-16 epochs, dropout slows convergence more than it regularizes. Model is compute-limited, not overfitting.
- **Simpler head (128->1 instead of 256->128->1)** improves by 0.31% — fewer params means faster epochs (16 vs 13). The conv output doesn't need a deep head.
- **max_lr=4e-3** (up from 3e-3) improves 0.10% — with more epochs from simpler architecture, slightly higher LR helps.
- **Virtual fixed-distance links FAILED badly.** Tried resampling link lookback at fixed distance intervals (10m and 80m per virtual link). Both 100m and 800m total lookback were 8-31% worse. Piecewise-constant interpolation creates redundant features (multiple virtual links from same actual link = constant values). Real link boundaries carry important transition information that resampling destroys.
- **64 conv channels** underfits — 128 is the right capacity even with simpler head.
- **SGD momentum** 119% worse than AdamW — far too slow to converge in 15 epochs. Adaptive LR essential.
- **OneCycleLR epochs must match actual training** — setting epochs=20 when training 15 keeps LR too high, hurting generalization.
- **Kernel size 5 first layer** nearly tied but marginally slower — with 3 layers of k=3, receptive field already covers seq_len=5.
- **Linear head (no hidden layer)** too simple — conv output needs at least one nonlinear combination.
- **LayerNorm, BatchNorm, Dropout1d in conv** all hurt — inputs are pre-normalized, model needs every epoch for convergence.
- **Channel attention (squeeze-and-excitation)** no benefit — too few channels/timesteps for attention to help.
- **Input noise augmentation** hurts — another regularization technique that needs many epochs to show benefit.
- **n_points (geometry vertex count)** no signal for CNN either (consistent with RF).
- **cum_backward_time** redundant with cum_backward_miles (correlated through speed).
- **Target normalization** no benefit — AdamW adapts to target scale.
- **torch.compile** incompatible with ProcessPoolExecutor (pickle error).
- **DataLoader num_workers>0** causes OneCycleLR step overflow inside ProcessPoolExecutor.

### Best known configuration (apr16/exp18, RMSE 0.006099)

- 1D-CNN: 3 Conv1d layers (128 channels, kernel_size=3, padding=1), ReLU activations
- Head: Linear(640, 128) -> ReLU -> Linear(128, 1) — no dropout
- Sequence: 5-link window (current + 4 previous), 7 features per link
- Link features: speed_mph, grade_percent, miles, time_seconds, sinuosity, abs_bearing_delta, cum_backward_miles
- No static features (link_position removed per domain constraint)
- StandardScaler normalization on sequence features
- AdamW optimizer, OneCycleLR max_lr=4e-3, weight_decay=1e-4, gradient clipping max_norm=1.0
- Batch size 2048, MSE loss
- ~16 epochs in 10-minute budget

### Best known RF configuration (apr13b/exp20, RMSE 0.006400)

- RandomForestRegressor with n_estimators=2000, max_depth=None, min_samples_split=10, max_features=0.7, max_samples=0.5
- Features: speed_mph, grade_percent, miles, prev_speed_mph, speed_delta, prev_miles, grade_delta, link_position, prev2_speed_mph, prev3_speed_mph, prev4_speed_mph, time_seconds, sinuosity, abs_bearing_delta, prev_sinuosity, prev_abs_bearing_delta, speed_accel, prev_time_seconds
- Geometry features parsed in a single pass from WKB LineString (SRID 4326)

## Cross-cutting insights

- RF consistently outperforms gradient boosting and extra-trees for this domain.
- Previous link speed features are far more valuable than previous link grade features. Speed encodes driver dynamics; grade is geography-driven.
- Pre-computed aggregations, deltas-of-deltas, absolute values, and interaction terms are all redundant for tree models — they can derive these from raw values. Only add derived features that encode information trees can't reach from the available columns (e.g., sinuosity requires parsing geometry).
- Subsampling (max_samples + max_features) is critical regularization for RF on this dataset.
- Speed lookback has diminishing returns: 1-link = huge, 2-link = large, 3-link = moderate, 4-link = small, 5-link = negative.
- Turn/curvature lookback saturates at 1 link, unlike speed.
- Geometry features worth extracting: sinuosity and bearing_delta. Shape-detail features (max_curvature, n_points) don't help because most road segments have only 2-4 geometry points.

## Open questions

- Would sub-link segmentation (splitting long links at geometry vertices) help?
- Is there value in treating different journey types differently (short urban vs long highway)?
- Would weather or time-of-day features help, if they could be joined in?
