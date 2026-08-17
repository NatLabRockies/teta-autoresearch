import json
import time
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, TimeoutError

from sklearn.ensemble import RandomForestRegressor


from fixed_utils import (
    evaluate,
    train_test_split,
)

# --- shared defaults ---
TIME_BUDGET_SECONDS = 10 * 60

LINK_FEATURES = [
    "speed_mph",
    "grade_percent",
    "miles",
]

TARGET = "energy_rate_gge"

# --- data config ---
DATA_PATH = "data/processed/2017_Chevy_Bolt.parquet"


def random_forest_model():
    model_params = {
        "n_estimators": 20,
        "max_depth": 10,
        "min_samples_split": 10,
        "random_state": 52,
        "n_jobs": -1,  # use all cores
    }
    model = RandomForestRegressor(**model_params)
    return model


def train_model() -> dict:
    """Train and evaluate. Returns results dict."""
    t0 = time.time()

    # data
    df = pd.read_parquet(DATA_PATH)

    # sort by journey and time
    df = df.sort_values(["journey_id", "link_start_time"])

    train_df, test_df = train_test_split(df, test_size=0.2, random_seed=42)

    y_train = train_df[TARGET].to_numpy(dtype=np.float32)
    y_test = test_df[TARGET].to_numpy(dtype=np.float32)
    journey_id_te = test_df["journey_id"].to_numpy()
    miles_te = test_df["miles"].to_numpy(dtype=np.float32)

    model = random_forest_model()
    model.fit(train_df[LINK_FEATURES], y_train)
    predicted = model.predict(test_df[LINK_FEATURES])

    results = evaluate(y_test, predicted, journey_id=journey_id_te, miles=miles_te)

    # One `name: value` line per metric for readability, then a single
    # machine-readable line that is the parse target. Do not hardcode metric
    # names here — whatever evaluate() returns is what gets reported.
    for k, v in results.items():
        print(f"{k}: {v:.6f}")
    print(f"metrics: {json.dumps({k: round(v, 6) for k, v in results.items()})}")

    # meta
    total_seconds = time.time() - t0
    print(f"total_seconds: {total_seconds:.1f}")
    print(f"features: {','.join(LINK_FEATURES)}")

    return results


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(train_model)
        try:
            future.result(timeout=TIME_BUDGET_SECONDS)
        except TimeoutError:
            print(f"\ntimed out after {TIME_BUDGET_SECONDS}s, skipping")
            executor.shutdown(wait=False, cancel_futures=True)
