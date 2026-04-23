"""Data-path resolution for the RouteE domain.

A tree is flattened (data at `<tree>/data/`); the template keeps data under
`domains/routee/data/`. Rather than hardcode one layout, walk up from this
file looking for a `data/` sibling.
"""

from pathlib import Path

ASSETS_BY_PARTITION: dict[str, str] = {
    "bev": "2017_Chevy_Bolt",
    "ice": "2016_Toyota_Camry",
    "phev": "TBD_PHEV",
}


def _find_data_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "data"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"could not locate a `data/` directory walking up from {here}"
    )


def default_data_path(partition: str) -> Path:
    asset = ASSETS_BY_PARTITION.get(partition)
    if asset is None:
        raise ValueError(
            f"unknown partition {partition!r}; valid: {sorted(ASSETS_BY_PARTITION)}"
        )
    return _find_data_root() / "processed" / f"{asset}.parquet"
