# %%
import pandas as pd

# %%
df = pd.read_parquet("data/raw/2017_Chevy_Bolt.parquet")
# %%
agg = (
    df.groupby(["journeyId", "road_id"], sort=False)
    .agg(
        secs=("time_gap", "sum"),
        miles=("simdrive_miles", "sum"),
        speed_mph=("speed_mph", "mean"),
        grade_dec=("grade_dec_filtered", "mean"),
        energy_gge=("ess_kwh_out_ach", "sum"),
    )
    .reset_index()
)
# %%
agg
# %%
# %%
agg[(agg["miles"] < 0.5)]["miles"].describe()
# %%
5.088465e-02 * 5280
# %%
