import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError

import pandas as pd
from sklearn.ensemble import RandomForestRegressor  # type: ignore[import-untyped]

from fixed_utils import (
    aggregate_links,
    filter_data,
    load_data,
    evaluate,
    train_test_split,
)

# --- shared defaults ---
TIME_BUDGET_SECONDS = 10 * 60
FEATURES = ["speed_mph", "grade_percent", "miles", "prev_speed_mph", "speed_delta", "prev_miles", "grade_delta", "link_position", "prev2_speed_mph", "prev3_speed_mph", "prev4_speed_mph", "time_seconds"]
TARGET = "energy_rate_gge"

# --- data config ---
CONFIG = {
    "name": "2017_Chevy_Bolt",
    "data_path": "data/raw/2017_Chevy_Bolt.parquet",
    "energy_type": "bev",
}


def train_model() -> dict:
    """Train and evaluate. Returns results dict."""
    t0 = time.time()

    # data
    df = load_data(CONFIG["data_path"], energy_type=CONFIG["energy_type"])
    df = aggregate_links(df)
    df = filter_data(df)
    # add previous link features (within each journey, ordered by link_start_time)
    df = df.sort_values(["journey_id", "link_start_time"])
    df["prev_speed_mph"] = df.groupby("journey_id")["speed_mph"].shift(1)
    df["speed_delta"] = df["speed_mph"] - df["prev_speed_mph"]
    df["prev_miles"] = df.groupby("journey_id")["miles"].shift(1)
    df["prev_grade_percent"] = df.groupby("journey_id")["grade_percent"].shift(1)
    df["grade_delta"] = df["grade_percent"] - df["prev_grade_percent"]
    df["link_position"] = df.groupby("journey_id").cumcount()
    df["prev2_speed_mph"] = df.groupby("journey_id")["speed_mph"].shift(2)
    df["prev3_speed_mph"] = df.groupby("journey_id")["speed_mph"].shift(3)
    df["prev4_speed_mph"] = df.groupby("journey_id")["speed_mph"].shift(4)
    df = df.dropna(subset=["prev_speed_mph", "prev2_speed_mph", "prev3_speed_mph", "prev4_speed_mph"])

    train_df, test_df = train_test_split(df, test_size=0.2, random_seed=42)

    # train
    X = train_df[FEATURES]
    y = train_df[TARGET]

    model_params = {
        "n_estimators": 2000,
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
