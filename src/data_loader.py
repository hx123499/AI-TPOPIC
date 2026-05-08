from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_trip_data(file_path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Load trip data from a parquet file."""
    return pd.read_parquet(file_path, columns=columns)


def get_required_columns() -> list[str]:
    """Return the columns used across preprocessing, analysis, and modeling."""
    return [
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "passenger_count",
        "trip_distance",
        "PULocationID",
        "DOLocationID",
        "fare_amount",
        "total_amount",
        "tip_amount",
        "payment_type",
    ]
