"""RouteE domain hooks for optimizer mode.

Contract (see EXTENDING.md): every domain that wants to run under
`optimizers/` ships a `search/` submodule that re-exports three callables:

    sample_config(trial, families=None) -> dict
    load_and_engineer(data_path: str) -> pd.DataFrame
    run_trial(config, df, budget_seconds, trial=None) -> float
"""

from .data import ASSETS_BY_PARTITION, default_data_path
from .feature_pipeline import load_and_engineer
from .models import run_trial
from .search_space import (
    ALL_FAMILIES,
    WARM_START_CONFIGS,
    sample_config,
    sample_config_phase1,
    sample_config_phase2,
)

__all__ = [
    "ALL_FAMILIES",
    "ASSETS_BY_PARTITION",
    "WARM_START_CONFIGS",
    "default_data_path",
    "load_and_engineer",
    "run_trial",
    "sample_config",
    "sample_config_phase1",
    "sample_config_phase2",
]
