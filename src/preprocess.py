from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import PROCESSED_DIR, REPORTS_DIR


NUMERIC_COLUMNS = [
    "passenger_count",
    "trip_distance",
    "fare_amount",
    "total_amount",
    "tip_amount",
]


def _iqr_outlier_count(series: pd.Series) -> int:
    """Count outliers using the IQR rule for a numeric series."""
    clean_series = series.dropna()
    if clean_series.empty:
        return 0

    q1 = clean_series.quantile(0.25)
    q3 = clean_series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return int(((series < lower_bound) | (series > upper_bound)).sum())


def build_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Generate and save a column-level data quality report."""
    report_rows: list[dict] = []

    duplicate_count = int(df.duplicated().sum())

    for column in df.columns:
        series = df[column]
        report_rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "missing_count": int(series.isna().sum()),
                "missing_rate": round(float(series.isna().mean()), 6),
                "unique_count": int(series.nunique(dropna=True)),
                "zero_count": int((series == 0).sum()) if pd.api.types.is_numeric_dtype(series) else np.nan,
                "negative_count": int((series < 0).sum()) if pd.api.types.is_numeric_dtype(series) else np.nan,
                "outlier_count_iqr": _iqr_outlier_count(series) if pd.api.types.is_numeric_dtype(series) else np.nan,
                "duplicate_rows_in_dataset": duplicate_count,
            }
        )

    report = pd.DataFrame(report_rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORTS_DIR / "quality_report.csv", index=False, encoding="utf-8-sig")
    return report


def clean_trip_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean trip records and save the cleaned dataset for later modules."""
    cleaned = df.copy()
    original_rows = len(cleaned)

    cleaned["tpep_pickup_datetime"] = pd.to_datetime(cleaned["tpep_pickup_datetime"], errors="coerce")
    cleaned["tpep_dropoff_datetime"] = pd.to_datetime(cleaned["tpep_dropoff_datetime"], errors="coerce")

    # Remove rows missing core business fields because later analysis and prediction depend on them.
    cleaned = cleaned.dropna(
        subset=[
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
            "PULocationID",
            "DOLocationID",
            "trip_distance",
            "fare_amount",
            "total_amount",
        ]
    )

    # Drop duplicated orders so repeated rows do not inflate demand counts.
    cleaned = cleaned.drop_duplicates()

    # Remove time-reversed trips because drop-off earlier than pickup is logically impossible.
    cleaned = cleaned[cleaned["tpep_dropoff_datetime"] >= cleaned["tpep_pickup_datetime"]]

    # Remove non-positive distance and fare rows because they break cost analysis and model reliability.
    cleaned = cleaned[(cleaned["trip_distance"] > 0) & (cleaned["fare_amount"] > 0) & (cleaned["total_amount"] > 0)]

    # Keep realistic passenger counts for yellow taxi trips; extreme values are usually data-entry noise.
    cleaned = cleaned[cleaned["passenger_count"].fillna(1).between(1, 6)]

    trip_duration_min = (
        (cleaned["tpep_dropoff_datetime"] - cleaned["tpep_pickup_datetime"]).dt.total_seconds() / 60
    )
    cleaned["trip_duration_min"] = trip_duration_min

    # Remove trips with zero/negative duration and very extreme duration to reduce anomaly distortion.
    cleaned = cleaned[(cleaned["trip_duration_min"] > 0) & (cleaned["trip_duration_min"] <= 180)]

    # Filter implausible numeric extremes using domain-informed caps for urban taxi trips.
    cleaned = cleaned[
        cleaned["trip_distance"].between(0.1, 80)
        & cleaned["fare_amount"].between(1, 300)
        & cleaned["total_amount"].between(1, 400)
    ]

    cleaned["passenger_count"] = cleaned["passenger_count"].fillna(cleaned["passenger_count"].median())
    cleaned["payment_type"] = cleaned["payment_type"].fillna(cleaned["payment_type"].mode().iloc[0])
    cleaned["tip_amount"] = cleaned["tip_amount"].fillna(0)

    cleaned = cleaned.reset_index(drop=True)

    summary = pd.DataFrame(
        [
            {
                "original_rows": original_rows,
                "cleaned_rows": len(cleaned),
                "removed_rows": original_rows - len(cleaned),
                "retention_rate": round(len(cleaned) / original_rows, 6),
            }
        ]
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(PROCESSED_DIR / "cleaned_tripdata_2023-01.parquet", index=False)
    summary.to_csv(REPORTS_DIR / "cleaning_summary.csv", index=False, encoding="utf-8-sig")
    return cleaned
