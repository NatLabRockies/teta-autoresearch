import pandas as pd
import numpy as np

from adjustable_utils import aggregate_links, filter_data

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

KWH_PER_GALLON_GASOLINE = 33.7

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


def prepare(
    path: str,
    energy_type: str,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, aggregate, filter, and split data into train/test sets."""
    df = load_data(path, energy_type)
    df = aggregate_links(df)
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
