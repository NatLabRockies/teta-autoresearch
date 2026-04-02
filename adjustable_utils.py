import pandas as pd

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
# Adjustable Data utilities (you can modify these)
# ---------------------------------------------------------------------------


def aggregate_links(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1Hz point-level data to link-level data.

    Groups by (journeyId, road_id) and computes link-level summaries.
    This is the step where raw simulation points become road-link records.
    """
    agg = (
        df.groupby(["journeyId", "road_id"], sort=False)
        .agg(
            secs=("time_gap", "sum"),
            miles=("simdrive_miles", "sum"),
            speed_mph=("speed_mph", "mean"),
            grade_dec=("grade_dec_filtered", "mean"),
            energy_gge=("energy_gge", "sum"),
            road_class=("road_class", "first"),
        )
        .reset_index()
    )
    agg = agg.rename(columns={"journeyId": "journey_id"})
    return agg


def filter_data(df: pd.DataFrame, energy_type: str) -> pd.DataFrame:
    """Filter outliers and engineer features for training."""
    df = df.copy()

    # drop road id of -1000
    df = df[df["road_id"] != -1000]

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
