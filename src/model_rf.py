from __future__ import annotations

import json

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from src.utils import REPORTS_DIR


def prepare_demand_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trip records into zone-hour demand samples for prediction."""
    demand_df = df.copy()
    demand_df = demand_df[demand_df["pickup_date"].str.startswith("2023-01")].copy()

    aggregated = (
        demand_df.groupby(["pickup_date", "pickup_hour", "PULocationID"])
        .agg(
            demand_count=("PULocationID", "size"),
            avg_distance=("trip_distance", "mean"),
            avg_fare=("fare_amount", "mean"),
            avg_duration=("trip_duration_min", "mean"),
            avg_speed=("speed_mph", "mean"),
            pickup_weekday=("pickup_weekday", "first"),
            is_weekend=("is_weekend", "first"),
            is_peak=("is_peak", "first"),
        )
        .reset_index()
        .sort_values(["pickup_date", "PULocationID", "pickup_hour"])
        .reset_index(drop=True)
    )

    aggregated["prev_hour_demand"] = aggregated.groupby(["pickup_date", "PULocationID"])["demand_count"].shift(1)
    aggregated["prev_hour_demand"] = aggregated["prev_hour_demand"].fillna(0)
    return aggregated


def build_model_features(demand_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Convert the aggregated demand table into model-ready features and labels."""
    feature_df = demand_df[
        [
            "pickup_hour",
            "PULocationID",
            "avg_distance",
            "avg_fare",
            "avg_duration",
            "avg_speed",
            "pickup_weekday",
            "is_weekend",
            "is_peak",
            "prev_hour_demand",
        ]
    ].copy()
    feature_df["PULocationID"] = feature_df["PULocationID"].astype(str)
    feature_df = pd.get_dummies(feature_df, columns=["PULocationID"], drop_first=False)
    target = demand_df["demand_count"].copy()
    return feature_df, target


def train_random_forest(df: pd.DataFrame) -> dict:
    """Train a random forest baseline for zone-hour demand prediction."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    demand_df = prepare_demand_dataset(df)
    features, target = build_model_features(demand_df)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=18,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, predictions))
    rmse = float(mean_squared_error(y_test, predictions) ** 0.5)

    feature_importance = (
        pd.DataFrame({"feature": X_train.columns, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(15)
    )
    feature_importance.to_csv(
        REPORTS_DIR / "random_forest_feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    result = {
        "model_name": "RandomForestRegressor",
        "mae": mae,
        "rmse": rmse,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "feature_count": int(X_train.shape[1]),
    }

    with open(REPORTS_DIR / "random_forest_metrics.json", "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    return result
