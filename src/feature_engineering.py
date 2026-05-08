from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import PROCESSED_DIR

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal and domain-derived features required by the assignment."""
    feature_df = df.copy()

    pickup_time = pd.to_datetime(feature_df["tpep_pickup_datetime"])

    feature_df["pickup_date"] = pickup_time.dt.date.astype(str)
    feature_df["pickup_hour"] = pickup_time.dt.hour
    feature_df["pickup_weekday"] = pickup_time.dt.dayofweek
    feature_df["is_weekend"] = feature_df["pickup_weekday"].isin([5, 6]).astype(int)
    feature_df["is_peak"] = feature_df["pickup_hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)

    # Derived feature 1: split the day into interpretable traffic periods for later grouped analysis.
    feature_df["time_period"] = pd.cut(
        feature_df["pickup_hour"],
        bins=[-1, 5, 11, 16, 20, 23],
        labels=["late_night", "morning", "afternoon", "evening", "night"],
    ).astype(str)

    # Derived feature 2: estimate trip speed to capture congestion-related travel efficiency.
    duration_hours = feature_df["trip_duration_min"] / 60
    feature_df["speed_mph"] = np.where(duration_hours > 0, feature_df["trip_distance"] / duration_hours, 0)
    feature_df["speed_mph"] = feature_df["speed_mph"].clip(lower=0, upper=80)

    # Derived feature 3: fare efficiency helps compare pricing differences across trips of different lengths.
    feature_df["fare_per_mile"] = feature_df["fare_amount"] / feature_df["trip_distance"]
    feature_df["fare_per_mile"] = feature_df["fare_per_mile"].replace([np.inf, -np.inf], np.nan).fillna(0)

    feature_df.to_parquet(PROCESSED_DIR / "featured_tripdata_2023-01.parquet", index=False)
    return feature_df
