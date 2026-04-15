"""Feature engineering pipeline.

Loads the parquet data and computes all candidate features used across
tabular and sequential model families. Call load_and_engineer() once at
process start; individual trials then select feature subsets from the result.
"""

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from shapely import wkb

# Make fixed_utils importable from the parent workspace
sys.path.insert(0, str(Path(__file__).parent.parent))
from fixed_utils import train_test_split  # noqa: E402

TARGET = "energy_rate_gge"

# Canonical ordered list of all engineered tabular features.
ALL_FEATURES: list[str] = [
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

# These three must always appear; never gated by a feature flag.
REQUIRED_FEATURES: set[str] = {"speed_mph", "grade_percent", "miles"}

# Per-timestep feature pool for sequential (CNN / GRU) models.
SEQ_FEATURE_POOL: list[str] = [
    "speed_mph",
    "grade_percent",
    "miles",
    "time_seconds",
    "sinuosity",
    "abs_bearing_delta",
]

# Per-link (non-sequenced) features appended after the recurrent/conv layers.
STATIC_FEATURE_POOL: list[str] = ["link_position"]


# ---------------------------------------------------------------------------
# Data loading + feature engineering
# ---------------------------------------------------------------------------


def load_and_engineer(data_path: str) -> pd.DataFrame:
    """Load parquet and compute all candidate features.

    Rows that cannot have full 4-link speed look-backs are dropped so every
    row in the returned frame has valid values for every feature in ALL_FEATURES.
    """
    df = pd.read_parquet(data_path)
    df = df.sort_values(["journey_id", "link_start_time"])

    # Lag / delta features
    df["prev_speed_mph"] = df.groupby("journey_id")["speed_mph"].shift(1)
    df["speed_delta"] = df["speed_mph"] - df["prev_speed_mph"]
    df["prev_miles"] = df.groupby("journey_id")["miles"].shift(1)
    prev_grade = df.groupby("journey_id")["grade_percent"].shift(1)
    df["grade_delta"] = df["grade_percent"] - prev_grade
    df["prev2_speed_mph"] = df.groupby("journey_id")["speed_mph"].shift(2)
    df["prev3_speed_mph"] = df.groupby("journey_id")["speed_mph"].shift(3)
    df["prev4_speed_mph"] = df.groupby("journey_id")["speed_mph"].shift(4)
    df["prev_time_seconds"] = df.groupby("journey_id")["time_seconds"].shift(1)
    df["link_position"] = df.groupby("journey_id").cumcount()
    prev_speed_delta = df.groupby("journey_id")["speed_delta"].shift(1)
    df["speed_accel"] = df["speed_delta"] - prev_speed_delta

    # Geometry features (sinuosity + bearing) — single pass over WKB LineStrings
    def _geom_feats(geom_hex: Any) -> tuple[float, float]:
        g = wkb.loads(geom_hex, hex=True)
        coords = list(g.coords)
        s, e = coords[0], coords[-1]
        dx, dy = e[0] - s[0], e[1] - s[1]
        straight = math.sqrt(dx**2 + dy**2)
        if straight < 1e-10:
            sinuosity = 1.0
        else:
            road_len = sum(
                math.sqrt(
                    (coords[i + 1][0] - coords[i][0]) ** 2
                    + (coords[i + 1][1] - coords[i][1]) ** 2
                )
                for i in range(len(coords) - 1)
            )
            sinuosity = road_len / straight
        bearing = math.atan2(dx, dy) * 180.0 / math.pi
        return sinuosity, bearing

    geom = df["geometry"].apply(_geom_feats)
    df["sinuosity"] = geom.apply(lambda x: x[0])
    df["bearing"] = geom.apply(lambda x: x[1])
    df["prev_sinuosity"] = df.groupby("journey_id")["sinuosity"].shift(1)
    prev_bearing = df.groupby("journey_id")["bearing"].shift(1)
    raw_delta = df["bearing"] - prev_bearing
    df["bearing_delta"] = (raw_delta + 180) % 360 - 180
    df["abs_bearing_delta"] = df["bearing_delta"].abs()
    df["prev_abs_bearing_delta"] = (
        df.groupby("journey_id")["abs_bearing_delta"].shift(1)
    )

    # Drop rows that are missing any deep lag (journey-boundary rows)
    df = df.dropna(
        subset=[
            "prev_speed_mph",
            "prev2_speed_mph",
            "prev3_speed_mph",
            "prev4_speed_mph",
            "bearing_delta",
        ]
    ).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Train / test splits
# ---------------------------------------------------------------------------


def tabular_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Row-level 80/20 split using fixed_utils (matches the LLM experiment harness)."""
    return train_test_split(df, test_size=0.2, random_seed=42)


def sequential_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Journey-level 80/20 split for sequential models.

    Splitting by journey prevents sequence windows from crossing the
    train/test boundary and leaking future information.
    """
    journey_ids = sorted(df["journey_id"].unique())
    n_test = max(1, int(len(journey_ids) * 0.2))
    test_ids = set(journey_ids[-n_test:])
    train_df = df[~df["journey_id"].isin(test_ids)].reset_index(drop=True)
    test_df = df[df["journey_id"].isin(test_ids)].reset_index(drop=True)
    return train_df, test_df


# ---------------------------------------------------------------------------
# Sequence window builder (CNN / GRU)
# ---------------------------------------------------------------------------


def build_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    static_cols: list[str],
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build fixed-length look-back windows, one per (journey, link) pair.

    Returns
    -------
    X_seq    : float32 array of shape (N, seq_len, len(feature_cols))
    X_static : float32 array of shape (N, len(static_cols))  — zeros if empty
    y        : float32 array of shape (N,)
    """
    seq_list: list[np.ndarray] = []
    static_list: list[np.ndarray] = []
    y_list: list[float] = []

    n_static = len(static_cols)
    for _, journey in df.groupby("journey_id"):
        journey = journey.reset_index(drop=True)
        n = len(journey)
        if n < seq_len:
            continue
        feats = journey[feature_cols].to_numpy(dtype=np.float32)
        statics = (
            journey[static_cols].to_numpy(dtype=np.float32)
            if n_static
            else np.zeros((n, 0), dtype=np.float32)
        )
        targets = journey[TARGET].to_numpy(dtype=np.float32)
        for i in range(seq_len - 1, n):
            seq_list.append(feats[i - seq_len + 1 : i + 1])
            static_list.append(statics[i])
            y_list.append(float(targets[i]))

    X_seq = np.stack(seq_list)  # (N, seq_len, F)
    X_static = (
        np.stack(static_list)
        if n_static
        else np.zeros((len(seq_list), 0), dtype=np.float32)
    )
    y = np.array(y_list, dtype=np.float32)
    return X_seq, X_static, y
