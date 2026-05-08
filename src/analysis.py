from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils import FIGURES_DIR

sns.set_theme(style="whitegrid")


def _save_figure(file_name: str) -> str:
    """Save the current matplotlib figure and return its path."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    file_path = FIGURES_DIR / file_name
    plt.tight_layout()
    plt.savefig(file_path, dpi=200, bbox_inches="tight")
    plt.close()
    return str(file_path)


def analyze_hourly_demand(df: pd.DataFrame) -> dict[str, str]:
    """Create time-based demand charts by hour and by weekday/weekend."""
    result: dict[str, str] = {}

    hourly_counts = df.groupby("pickup_hour").size().reset_index(name="order_count")
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=hourly_counts, x="pickup_hour", y="order_count", marker="o", color="#1f77b4")
    plt.title("Hourly Travel Demand")
    plt.xlabel("Pickup Hour")
    plt.ylabel("Order Count")
    result["hourly_demand"] = _save_figure("hourly_demand.png")

    by_day_type = (
        df.groupby(["pickup_hour", "is_weekend"])
        .size()
        .reset_index(name="order_count")
        .replace({"is_weekend": {0: "Weekday", 1: "Weekend"}})
    )
    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=by_day_type,
        x="pickup_hour",
        y="order_count",
        hue="is_weekend",
        marker="o",
        palette=["#2a9d8f", "#e76f51"],
    )
    plt.title("Hourly Demand: Weekday vs Weekend")
    plt.xlabel("Pickup Hour")
    plt.ylabel("Order Count")
    plt.legend(title="Day Type")
    result["hourly_daytype_demand"] = _save_figure("hourly_weekday_weekend_demand.png")

    return result


def analyze_region_heat(df: pd.DataFrame) -> dict[str, str]:
    """Create top-region and peak-hour distribution charts."""
    result: dict[str, str] = {}

    top_pickup = df["PULocationID"].value_counts().head(10).sort_values()
    plt.figure(figsize=(9, 6))
    sns.barplot(x=top_pickup.values, y=top_pickup.index.astype(str), color="#457b9d")
    plt.title("Top 10 Pickup Zones")
    plt.xlabel("Order Count")
    plt.ylabel("PULocationID")
    result["top_pickup_zones"] = _save_figure("top10_pickup_zones.png")

    top_dropoff = df["DOLocationID"].value_counts().head(10).sort_values()
    plt.figure(figsize=(9, 6))
    sns.barplot(x=top_dropoff.values, y=top_dropoff.index.astype(str), color="#f4a261")
    plt.title("Top 10 Dropoff Zones")
    plt.xlabel("Order Count")
    plt.ylabel("DOLocationID")
    result["top_dropoff_zones"] = _save_figure("top10_dropoff_zones.png")

    top_zone_ids = df["PULocationID"].value_counts().head(10).index.tolist()
    heatmap_df = (
        df[df["PULocationID"].isin(top_zone_ids)]
        .groupby(["PULocationID", "pickup_hour"])
        .size()
        .reset_index(name="order_count")
        .pivot(index="PULocationID", columns="pickup_hour", values="order_count")
        .fillna(0)
    )
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_df, cmap="YlOrRd")
    plt.title("Peak-hour Distribution of Top Pickup Zones")
    plt.xlabel("Pickup Hour")
    plt.ylabel("PULocationID")
    result["pickup_zone_heatmap"] = _save_figure("pickup_zone_hour_heatmap.png")

    return result


def analyze_fare_factors(df: pd.DataFrame) -> dict[str, str]:
    """Analyze how distance, time, and passengers relate to fare."""
    result: dict[str, str] = {}

    sample_df = df.sample(n=min(20000, len(df)), random_state=42)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=sample_df,
        x="trip_distance",
        y="fare_amount",
        alpha=0.35,
        s=20,
        color="#264653",
        edgecolor=None,
    )
    plt.title("Trip Distance vs Fare Amount")
    plt.xlabel("Trip Distance (miles)")
    plt.ylabel("Fare Amount ($)")
    result["distance_fare_scatter"] = _save_figure("distance_fare_scatter.png")

    plt.figure(figsize=(9, 5))
    fare_by_period = sample_df.copy()
    sns.boxplot(
        data=fare_by_period,
        x="time_period",
        y="fare_amount",
        order=["late_night", "morning", "afternoon", "evening", "night"],
        color="#8ecae6",
    )
    plt.title("Fare Distribution by Time Period")
    plt.xlabel("Time Period")
    plt.ylabel("Fare Amount ($)")
    result["fare_by_time_period"] = _save_figure("fare_by_time_period_boxplot.png")

    passenger_fare = (
        df.groupby("passenger_count")["fare_amount"]
        .mean()
        .reset_index()
        .sort_values("passenger_count")
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(data=passenger_fare, x="passenger_count", y="fare_amount", color="#2a9d8f")
    plt.title("Average Fare by Passenger Count")
    plt.xlabel("Passenger Count")
    plt.ylabel("Average Fare ($)")
    result["fare_by_passenger_count"] = _save_figure("fare_by_passenger_count.png")

    return result


def analyze_congestion_insight(df: pd.DataFrame) -> dict[str, str]:
    """Create an insight-focused chart around congestion and travel efficiency."""
    result: dict[str, str] = {}

    congestion_df = (
        df.groupby("pickup_hour")[["speed_mph", "trip_duration_min", "fare_amount"]]
        .mean()
        .reset_index()
    )

    fig, ax1 = plt.subplots(figsize=(11, 6))
    sns.lineplot(data=congestion_df, x="pickup_hour", y="speed_mph", marker="o", color="#1d3557", ax=ax1)
    ax1.set_xlabel("Pickup Hour")
    ax1.set_ylabel("Average Speed (mph)", color="#1d3557")
    ax1.tick_params(axis="y", labelcolor="#1d3557")

    ax2 = ax1.twinx()
    sns.lineplot(
        data=congestion_df,
        x="pickup_hour",
        y="trip_duration_min",
        marker="s",
        color="#e63946",
        ax=ax2,
    )
    ax2.set_ylabel("Average Trip Duration (min)", color="#e63946")
    ax2.tick_params(axis="y", labelcolor="#e63946")
    plt.title("Congestion Insight: Speed and Duration by Hour")
    result["congestion_insight"] = _save_figure("congestion_speed_duration.png")

    return result


def run_all_analyses(df: pd.DataFrame) -> dict[str, str]:
    """Run all assignment-required analyses and return saved chart paths."""
    chart_paths: dict[str, str] = {}
    for analysis_result in [
        analyze_hourly_demand(df),
        analyze_region_heat(df),
        analyze_fare_factors(df),
        analyze_congestion_insight(df),
    ]:
        chart_paths.update(analysis_result)
    return chart_paths
