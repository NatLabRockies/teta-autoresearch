# Learnings

Accumulated findings across experiment sessions. Read this at the start of every new session to avoid repeating dead ends and to build on what works.

## 2017 Chevy Bolt (BEV)

### What works

- **Previous link speed features are the single biggest lever.** prev_speed_mph gave 24% improvement. Adding speed_delta (current - prev speed) gave another 7.4%. Deeper lookbacks (prev2, prev3, prev4) each added 2-5%. These capture acceleration/deceleration dynamics between links.
- **prev_miles** — previous link distance provides road-type transition context. 2.8% improvement.
- **grade_delta** — grade change between consecutive links captures elevation transition dynamics. 0.34% improvement. Tested removing it later — RMSE got worse, so it carries real signal.
- **link_position** — cumulative link count within a journey captures warm-up/cold-start effects. 0.17% improvement. Tested removing it — worse without it.
- **time_seconds** — time to traverse the link. 0.1% improvement. Tested removing it — worse without it. Captures something beyond miles/speed ratio.
- **max_samples=0.5** — subsampling acts as regularization and speeds up training. 0.41% improvement over no subsampling.
- **max_features=0.7** — feature subsampling per split increases tree diversity. 0.38% improvement.
- **max_depth=None (unlimited)** — with subsampling regularization, unlimited depth doesn't overfit and captures deeper interactions.
- **n_estimators=2000** — diminishing returns but marginal gains. Training time ~511s, close to budget.

### What doesn't work

- **speed_squared** — RF already learns nonlinear speed relationships via tree splits. Polynomial features are redundant for tree models.
- **prev_grade_percent** — grade history consistently adds no signal. Grade is geography-dependent, not driver-dependent. Tested multiple times (exp3, exp19, exp30) — never helps.
- **prev5_speed_mph (5-link lookback)** — signal plateaus at 4 links. 5th lookback hurts due to noise and data loss from dropna on short journeys.
- **speed_rolling_mean4** — RF can compute mean-like aggregations from individual values. Pre-computed aggregations are redundant for tree models.
- **speed_std4** — same as above, redundant for tree models.
- **cumulative_miles** — highly correlated with link_position. Adds no new signal.
- **HistGradientBoostingRegressor** — 1.3% worse than RF. RF's bagging approach handles the noisy energy data better than sequential boosting.
- **min_samples_split=5** — no improvement over 10, just slower training.
- **max_samples=0.3** — too aggressive, loses too much signal per tree. 0.5 is the sweet spot.
- **n_estimators=400** — negligible improvement over 200 but doubles training time. Only worth increasing with subsampling enabled.
- **Predicting energy_gge instead of rate** — 1.55% worse. Division by miles amplifies errors on short links. Direct rate prediction is better.

### Best known configuration (apr13/exp33, RMSE 0.007984)

- RandomForestRegressor with n_estimators=2000, max_depth=None, min_samples_split=10, max_features=0.7, max_samples=0.5
- Features: speed_mph, grade_percent, miles, prev_speed_mph, speed_delta, prev_miles, grade_delta, link_position, prev2_speed_mph, prev3_speed_mph, prev4_speed_mph, time_seconds
- Training time ~511s

## Cross-cutting insights

- RF consistently outperforms gradient boosting for this domain.
- Previous link speed features are far more valuable than previous link grade features. Speed encodes driver dynamics; grade is geography-driven.
- Pre-computed aggregations (rolling means, std) are redundant for tree models — they can derive these from raw values.
- Subsampling (max_samples + max_features) is critical regularization for RF on this dataset.
- Speed lookback has diminishing returns: 1-link = huge, 2-link = large, 3-link = moderate, 4-link = small, 5-link = negative.

## Open questions

- Are there other feature engineering approaches beyond speed lookback that could break below 0.008 RMSE?
- Would interaction features (e.g., speed * grade) help, or does RF handle these naturally?
- Is there value in treating different journey types differently (short urban vs long highway)?
