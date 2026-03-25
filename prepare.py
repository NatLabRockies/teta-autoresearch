import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

KWH_PER_GALLON_GASOLINE = 33.7
ALL_COLUMNS = [
    "journey_id",  # the trip id
    "road_id",  # the link id
    "time_seconds",  # time spent on the link in seconds
    "speed_mph",  # average speed on the link in miles per hour
    "grade_percent",  # average grade on the link in percent (i.e. -0.05 means 5% downhill, 0.1 means 10% uphill)
    "road_class",  # the class of the road (e.g. 1 = highway, 2 = primary, etc.)
    "miles",  # length of the link in miles
    "energy_gge",  # energy consumed on the link in gallons of gasoline equivalent (gge)
    "energy_rate_gge",  # energy rate on the link in gge per mile
]


# ---------------------------------------------------------------------------
# Data utilities (fixed, do not modify)
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


def filter_data(df: pd.DataFrame, energy_type: str) -> pd.DataFrame:
    """Filter outliers and engineer features for training."""
    df = df.copy()

    df = df.rename(columns={"secs": "time_seconds"})  # standardize column names

    # Remove zero-distance links to avoid division by zero
    df = df[df["miles"] > 0]

    # Compute energy rate
    df["energy_rate_gge"] = df["energy_gge"] / df["miles"]

    # Filter extreme energy rates
    df = df[df["energy_rate_gge"] < 0.4]

    # Filter extreme speeds
    df = df[df["speed_mph"] <= 120]

    # Convert grade from decimal to percent
    df["grade_percent"] = df["grade_dec"] * 100

    # Filter extreme grades
    df = df[df["grade_percent"].between(-20, 20)]

    df = df[ALL_COLUMNS]  # keep only relevant columns

    # Drop rows with NaN
    df = df.dropna()

    return df


def prepare(
    path: str,
    energy_type: str,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, aggregate, filter, and split data into train/test sets."""
    df = load_data(path, energy_type)
    df = filter_data(df, energy_type)

    # Train/test split
    rng = np.random.default_rng(random_seed)
    mask = rng.random(len(df)) < test_size
    test_df = df[mask].reset_index(drop=True)
    train_df = df[~mask].reset_index(drop=True)

    return train_df, test_df


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
