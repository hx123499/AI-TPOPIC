from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_trip_data(file_path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Load trip data from a parquet file."""
    return pd.read_parquet(file_path, columns=columns)
