# Data

The datasets are **not committed** to this repository — `data/raw/` and `data/processed/` are
gitignored. `train.py` reads:

```
data/processed/2017_Chevy_Bolt.parquet
```

You need to put a file there before a session can run.

## What the reference dataset is

Simulated 1 Hz drive-cycle traces for a 2017 Chevy Bolt, aggregated from point-level vehicle
dynamics up to the road-segment ("link") level. One row per link.

| column            | meaning                                                    |
| ----------------- | ---------------------------------------------------------- |
| `journey_id`      | identifies the trip a link belongs to                      |
| `link_start_time` | orders links within a journey                              |
| `speed_mph`       | average speed over the link                                |
| `grade_percent`   | average road gradient over the link                        |
| `miles`           | link distance                                              |
| `time_seconds`    | link traversal time                                        |
| `geometry`        | link geometry, WKB, EPSG:4326                              |
| `energy_rate_gge` | **target** — energy per mile in gasoline-gallon-equivalent |

`energy_rate_gge` can be negative for a BEV (regenerative braking).

`journey_id` and `miles` are load-bearing beyond being features: `fixed_utils.evaluate()` needs
both to compute trip-level totals.

## Using your own data

Any parquet with the columns above will work. At minimum you need `journey_id`, `miles`, and the
target; the rest are features `train.py` happens to start with. If your problem has a different
shape, change `fixed_utils.evaluate()` to define the metrics that matter and update `domain.md`
to describe the constraints — the protocol in `program.md` follows from those two files and needs
no edits.
