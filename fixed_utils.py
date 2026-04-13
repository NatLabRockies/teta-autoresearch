import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

KWH_PER_GALLON_GASOLINE = 33.7
FEET_PER_MILE = 5280
SHORT_LINK_THRESHOLD_FEET = 10
LONG_LINK_THRESHOLD_MILES = 0.5

# ---------------------------------------------------------------------------
# Fixed Data utilities (do not modify)
# ---------------------------------------------------------------------------


def load_data(path: str, energy_type: str) -> pd.DataFrame:
    """Load parquet data and add a unified 'energy' column."""
    df = pd.read_parquet(path)
    if energy_type == "bev":
        df["energy_gge"] = df["ess_kwh_out_ach"] / KWH_PER_GALLON_GASOLINE
    elif energy_type == "ice":
        df["energy_gge"] = df["fs_kwh_out_ach"] / KWH_PER_GALLON_GASOLINE
    else:
        raise ValueError(f"Unknown energy_type: {energy_type}. Use 'bev' or 'ice'.")
    return df


def train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train/test sets."""
    rng = np.random.default_rng(random_seed)
    mask = rng.random(len(df)) < test_size
    test_df = df[mask].reset_index(drop=True)
    train_df = df[~mask].reset_index(drop=True)
    return train_df, test_df


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter outliers and clean data."""
    df = df.copy()

    # drop road id of -1000 which indicates a map matching failure.
    df = df[df["road_id"] != -1000]

    df = df.rename(columns={"secs": "time_seconds"})  # standardize column names

    # Remove very short links that we wouldn't expect to see in practice
    distance_threhold_miels = SHORT_LINK_THRESHOLD_FEET / FEET_PER_MILE
    df = df[df["miles"] > distance_threhold_miels]

    # Remove very long links that are likely data errors
    df = df[df["miles"] < LONG_LINK_THRESHOLD_MILES]

    # Compute energy rate
    df["energy_rate_gge"] = df["energy_gge"] / df["miles"]

    # Filter extreme energy rates
    df = df[(df["energy_rate_gge"] < 0.4) & (df["energy_rate_gge"] > -0.4)]

    # Filter extreme speeds
    df = df[df["speed_mph"] <= 120]

    # Convert grade from decimal to percent
    df["grade_percent"] = df["grade_dec"] * 100

    # Filter extreme grades
    df = df[df["grade_percent"].between(-20, 20)]

    # Drop rows with NaN
    df = df.dropna()

    return df


# ---------------------------------------------------------------------------
# Evaluation Metrics (fixed, do not modify)
# ---------------------------------------------------------------------------


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def evaluate(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Evaluate a trained model on test data and return all metrics."""
    return {
        "rmse": rmse(actual, predicted),
    }
