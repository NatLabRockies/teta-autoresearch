# %%
from pathlib import Path

import pandas as pd

# %%
KWH_PER_GALLON_GASOLINE = 33.7
FEET_PER_MILE = 5280
SHORT_LINK_THRESHOLD_FEET = 10
LONG_LINK_THRESHOLD_MILES = 0.5


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
    df = df[(df["energy_rate_gge"] < 0.1) & (df["energy_rate_gge"] > -0.1)]

    # Filter extreme speeds
    df = df[df["speed_mph"] <= 120]

    # Convert grade from decimal to percent
    df["grade_percent"] = df["grade_dec"] * 100

    # Filter extreme grades
    df = df[df["grade_percent"].between(-20, 20)]

    # Drop links with less than 2 points per link
    df = df[df["n_points"] >= 2]

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
    else:
        raise ValueError(f"Unknown energy_type: {energy_type}. Use 'bev' or 'ice'.")
    return df


# %%
data_paths = ["raw/2017_Chevy_Bolt.parquet", "raw/2016_Toyota_Camry.parquet"]
# %%
# %%
for p in data_paths:
    data_path = Path(p)
    df = load_data(data_path, energy_type="bev")
    df = aggregate_links(df)
    df = filter_data(df)
    df.to_parquet(f"processed/{data_path.stem}.parquet", index=False)
# %%
df
# %%
# %%
