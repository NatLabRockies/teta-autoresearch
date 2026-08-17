import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Fixed Data utilities (do not modify)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Evaluation Metrics (fixed, do not modify)
# ---------------------------------------------------------------------------
#
# `evaluate` is the single place metrics are named. Everything downstream
# follows from the keys of the dict it returns: the lines train.py prints,
# the metric columns in the results TSV, the per-metric fields in the
# JSONL, and the Pareto keep rule in program.md. To research a different
# problem, change this function — the protocol needs no edits.


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Link-level root mean squared error on energy_rate_gge (GGE/mile)."""
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def trip_rmse(
    actual: np.ndarray,
    predicted: np.ndarray,
    journey_id: np.ndarray,
    miles: np.ndarray,
) -> float:
    """Trip total-energy RMSE in GGE.

    For each trip j: trip_gge = sum_i(rate_i * miles_i) over the links of j.
    Returns sqrt(mean((actual_trip_gge - pred_trip_gge)^2)) across trips.
    """
    df = pd.DataFrame(
        {
            "journey_id": journey_id,
            "actual_gge": np.asarray(actual) * np.asarray(miles),
            "pred_gge": np.asarray(predicted) * np.asarray(miles),
        }
    )
    trip = df.groupby("journey_id", sort=False)[["actual_gge", "pred_gge"]].sum()
    diff = trip["actual_gge"].to_numpy() - trip["pred_gge"].to_numpy()
    return float(np.sqrt(np.mean(diff**2)))


def evaluate(
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    journey_id: np.ndarray,
    miles: np.ndarray,
) -> dict[str, float]:
    """Evaluate a trained model and return both link-level and trip-level RMSE.

    `journey_id` and `miles` must align 1:1 with `actual` / `predicted`.
    """
    return {
        "rmse": rmse(actual, predicted),
        "trip_rmse": trip_rmse(actual, predicted, journey_id, miles),
    }
