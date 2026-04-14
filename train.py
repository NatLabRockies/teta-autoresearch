import time
import math
import pandas as pd
from shapely import wkb
from concurrent.futures import ProcessPoolExecutor, TimeoutError

from sklearn.ensemble import RandomForestRegressor  # type: ignore[import-untyped]

from fixed_utils import (
    evaluate,
    train_test_split,
)

# --- shared defaults ---
TIME_BUDGET_SECONDS = 10 * 60
FEATURES = [
    "speed_mph",
    "grade_percent",
    "miles",
    "prev_speed_mph",
    "speed_delta",
    "prev_miles",
    "grade_delta",
    "prev2_speed_mph",
    "prev3_speed_mph",
    "prev4_speed_mph",
    "time_seconds",
    "sinuosity",
    "abs_bearing_delta",
    "link_position",
    "prev_sinuosity",
    "prev_abs_bearing_delta",
    "speed_accel",
    "prev_time_seconds",
]
TARGET = "energy_rate_gge"

# --- data config ---
CONFIG = {
    "name": "2017_Chevy_Bolt",
    "data_path": "data/processed/2017_Chevy_Bolt.parquet",
    "energy_type": "bev",
}


def train_model() -> dict:
    """Train and evaluate. Returns results dict."""
    t0 = time.time()

    # data
    df = pd.read_parquet(CONFIG["data_path"])

    # add previous link features (within each journey, ordered by link_start_time)
    df = df.sort_values(["journey_id", "link_start_time"])
    df["prev_speed_mph"] = df.groupby("journey_id")["speed_mph"].shift(1)
    df["speed_delta"] = df["speed_mph"] - df["prev_speed_mph"]
    df["prev_miles"] = df.groupby("journey_id")["miles"].shift(1)
    df["prev_grade_percent"] = df.groupby("journey_id")["grade_percent"].shift(1)
    df["grade_delta"] = df["grade_percent"] - df["prev_grade_percent"]
    df["prev2_speed_mph"] = df.groupby("journey_id")["speed_mph"].shift(2)
    df["prev3_speed_mph"] = df.groupby("journey_id")["speed_mph"].shift(3)
    df["prev4_speed_mph"] = df.groupby("journey_id")["speed_mph"].shift(4)
    df["prev_time_seconds"] = df.groupby("journey_id")["time_seconds"].shift(1)
    df["link_position"] = df.groupby("journey_id").cumcount()
    prev_speed_delta = df.groupby("journey_id")["speed_delta"].shift(1)
    df["speed_accel"] = df["speed_delta"] - prev_speed_delta

    # geometry: extract sinuosity and bearing in single pass
    def calc_geom_features(geom_hex):
        g = wkb.loads(geom_hex, hex=True)
        coords = list(g.coords)
        start, end = coords[0], coords[-1]
        dx, dy = end[0] - start[0], end[1] - start[1]
        straight = math.sqrt(dx**2 + dy**2)
        if straight < 1e-10:
            sinuosity = 1.0
        else:
            road_len = sum(
                math.sqrt((coords[i+1][0]-coords[i][0])**2 + (coords[i+1][1]-coords[i][1])**2)
                for i in range(len(coords)-1)
            )
            sinuosity = road_len / straight
        bearing = math.atan2(dx, dy) * 180 / math.pi
        return sinuosity, bearing

    geom_feats = df["geometry"].apply(calc_geom_features)
    df["sinuosity"] = geom_feats.apply(lambda x: x[0])
    df["bearing"] = geom_feats.apply(lambda x: x[1])
    df["prev_sinuosity"] = df.groupby("journey_id")["sinuosity"].shift(1)
    df["prev_bearing"] = df.groupby("journey_id")["bearing"].shift(1)
    # Normalize bearing delta to [-180, 180]
    raw_delta = df["bearing"] - df["prev_bearing"]
    df["bearing_delta"] = (raw_delta + 180) % 360 - 180
    df["abs_bearing_delta"] = df["bearing_delta"].abs()
    df["prev_abs_bearing_delta"] = df.groupby("journey_id")["abs_bearing_delta"].shift(1)

    df = df.dropna(
        subset=[
            "prev_speed_mph",
            "prev2_speed_mph",
            "prev3_speed_mph",
            "prev4_speed_mph",
            "bearing_delta",
        ]
    )

    train_df, test_df = train_test_split(df, test_size=0.2, random_seed=42)

    # train
    X = train_df[FEATURES]
    y = train_df[TARGET]

    model_params = {
        "n_estimators": 1000,
        "max_depth": None,
        "min_samples_split": 10,
        "max_features": 0.7,
        "max_samples": 0.5,
        "random_state": 52,
        "n_jobs": -1,  # use all cores
    }

    model = RandomForestRegressor(**model_params)
    model.fit(X, y)

    # evaluate
    actual = test_df[TARGET].to_numpy()
    predicted = model.predict(test_df[FEATURES])
    results = evaluate(actual, predicted)
    for k, v in results.items():
        print(f"{k}: {v:.6f}")

    # meta
    total_seconds = time.time() - t0
    print(f"total_seconds: {total_seconds:.1f}")
    print(f"features: {','.join(FEATURES)}")

    return results


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(train_model)
        try:
            future.result(timeout=TIME_BUDGET_SECONDS)
        except TimeoutError:
            print(
                f"\n{CONFIG['name']}: timed out after {TIME_BUDGET_SECONDS}s, skipping"
            )
            executor.shutdown(wait=False, cancel_futures=True)
