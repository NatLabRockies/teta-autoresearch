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
FEATURES = ["speed_mph", "grade_percent", "miles"]
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
    train_df, test_df = train_test_split(df, test_size=0.2, random_seed=42)

    # train
    X = train_df[FEATURES]
    y = train_df[TARGET]

    model_params = {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 10,
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
