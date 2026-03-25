import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import torch

from prepare import prepare, evaluate

# --- shared defaults ---
TIME_BUDGET_SECONDS = 5 * 60
FEATURES = ["speed_mph", "grade_percent"]
TARGET = "energy_rate_gge"

# --- vehicle configs ---
# Each vehicle has its own data path, energy type, and model hyperparameters.
# This makes it easy to swap in different architectures per vehicle later.
VEHICLES = {
    "2017_Chevy_Bolt": {
        "data_path": "data/raw/2017_Chevy_Bolt.parquet",
        "energy_type": "bev",
        "model": {
            "n_estimators": 20,
            "max_depth": 10,
            "min_samples_split": 10,
            "random_state": 52,
            "n_jobs": 4,
        },
    },
    "2016_Toyota_Camry": {
        "data_path": "data/raw/2016_Toyota_Camry.parquet",
        "energy_type": "ice",
        "model": {
            "n_estimators": 20,
            "max_depth": 10,
            "min_samples_split": 10,
            "random_state": 52,
            "n_jobs": 4,
        },
    },
}


def train_random_forest(
    train_df: pd.DataFrame,
    features: list[str],
    target: str,
    model_params: dict,
) -> RandomForestRegressor:
    """Train a RandomForestRegressor on the given data."""
    X = train_df[features]
    y = train_df[target]

    model = RandomForestRegressor(**model_params)
    model.fit(X, y)

    return model


def run_vehicle(name: str, config: dict) -> dict:
    """Train and evaluate a single vehicle. Returns results dict."""
    t0 = time.time()
    print(f"\n{'=' * 40}")
    print(f"vehicle: {name}")
    print(f"{'=' * 40}")

    # data
    train_df, test_df = prepare(config["data_path"], energy_type=config["energy_type"])

    # train
    model = train_random_forest(train_df, FEATURES, TARGET, config["model"])

    # evaluate
    actual = test_df[TARGET].values
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
    for vehicle_name, vehicle_config in VEHICLES.items():
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_vehicle, vehicle_name, vehicle_config)
            try:
                future.result(timeout=TIME_BUDGET_SECONDS)
            except TimeoutError:
                print(f"\n{vehicle_name}: timed out after {TIME_BUDGET_SECONDS}s, skipping")
                executor.shutdown(wait=False, cancel_futures=True)
