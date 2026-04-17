# %%
from pathlib import Path

import pandas as pd

# %%
KWH_PER_GALLON_GASOLINE = 33.7
FEET_PER_MILE = 5280
SPEED_MPH_UPPER_BOUND = 120
GRADE_PERCENT_UPPER_BOUND = 20
GRADE_PERCENT_LOWER_BOUND = -20
SHORT_LINK_THRESHOLD_FEET = 10
LONG_LINK_THRESHOLD_MILES = 0.5
GGE_PER_MILE_LOWER_BOUND = -0.3
GGE_PER_MILE_UPPER_BOUND = 0.3
N_POINTS_LOWER_BOUND = 2
N_POINTS_UPPER_BOUND = 100


# %%
def aggregate_links(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1Hz point-level data to link-level data.

    Groups by (journeyId, road_id) and computes link-level summaries.
    This is the step where raw simulation points become road-link records.
    """
    agg = (
        df.groupby(["journeyId", "road_id"], sort=False)
        .agg(
            secs=("time_gap", "sum"),
            link_start_time=("time_rel", "min"),
            link_end_time=("time_rel", "max"),
            miles=("simdrive_miles", "sum"),
            speed_mph=("speed_mph", "mean"),
            grade_dec=("grade_dec_filtered", "mean"),
            energy_gge=("energy_gge", "sum"),
            n_points=("datapointId", "count"),
            geometry=("geom", "first"),
        )
        .reset_index()
    )
    agg = agg.rename(columns={"journeyId": "journey_id"})
    return agg


def filter_and_clean(df: pd.DataFrame) -> pd.DataFrame:
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
    df = df[
        (df["energy_rate_gge"] < GGE_PER_MILE_UPPER_BOUND)
        & (df["energy_rate_gge"] > GGE_PER_MILE_LOWER_BOUND)
    ]

    # Filter extreme speeds
    df = df[df["speed_mph"] <= SPEED_MPH_UPPER_BOUND]

    # Convert grade from decimal to percent
    df["grade_percent"] = df["grade_dec"] * 100

    # Filter extreme grades
    df = df[
        df["grade_percent"].between(
            GRADE_PERCENT_LOWER_BOUND, GRADE_PERCENT_UPPER_BOUND
        )
    ]

    # Drop links with less than N_POINTS_LOWER_BOUND points per link
    # or more than N_POINT_UPPER_BOUND points per link
    df = df[
        (df["n_points"] >= N_POINTS_LOWER_BOUND)
        & (df["n_points"] <= N_POINTS_UPPER_BOUND)
    ]

    # Drop rows with NaN
    df = df.dropna()

    return df


# %%
def load_data(path: str | Path, energy_type: str) -> pd.DataFrame:
    """Load parquet data and add a unified 'energy' column."""
    df = pd.read_parquet(path)
    if energy_type == "bev":
        df["energy_gge"] = df["ess_kwh_out_ach"] / KWH_PER_GALLON_GASOLINE
    elif energy_type == "ice":
        df["energy_gge"] = df["fs_kwh_out_ach"] / KWH_PER_GALLON_GASOLINE
    elif energy_type == "phev":
        df["energy_gge"] = (
            df["ess_kwh_out_ach"] + df["fs_kwh_out_ach"]
        ) / KWH_PER_GALLON_GASOLINE
    else:
        raise ValueError(
            f"Unknown energy_type: {energy_type}. Use 'bev', 'ice', or 'phev'."
        )
    return df


# %%
# Map each raw file stem to its powertrain / energy_type so each file is
# processed with the correct energy accounting.
POWERTRAIN_MAP = {
    "2017_Chevy_Bolt": "bev",
    "2016_Toyota_Camry": "ice",
    # Add PHEV entry (stem -> "phev") once the raw data lands.
}

data_paths = ["raw/2017_Chevy_Bolt.parquet", "raw/2016_Toyota_Camry.parquet"]
# %%
# %%
for p in data_paths:
    data_path = Path(p)
    energy_type = POWERTRAIN_MAP[data_path.stem]
    df = load_data(data_path, energy_type=energy_type)
    df = aggregate_links(df)
    df = filter_and_clean(df)
    df.to_parquet(f"processed/{data_path.stem}.parquet", index=False)
# %%
df["energy_rate_gge"].hist(bins=50)
# %%
bev = pd.read_parquet("processed/2017_Chevy_Bolt.parquet")
# %%
bev["n_points"].hist(bins=50)
# %%
