from __future__ import annotations

import re

import pandas as pd


def _extract_hour(question: str) -> int | None:
    match = re.search(r"(\d{1,2})\s*点", question)
    if not match:
        match = re.search(r"hour\s*(\d{1,2})", question.lower())
    if match:
        hour = int(match.group(1))
        if 0 <= hour <= 23:
            return hour
    return None


def _extract_zone_id(question: str) -> int | None:
    match = re.search(r"(?:区域|zone|Zone|PULocationID)\s*(\d+)", question)
    if match:
        return int(match.group(1))
    return None


def answer_hourly_demand(question: str, df: pd.DataFrame, chart_paths: dict[str, str]) -> str:
    hour = _extract_hour(question)
    if hour is None:
        return "请在问题中包含具体小时，例如“18点需求多少”。"

    demand = int((df["pickup_hour"] == hour).sum())
    return (
        f"{hour}点的订单量为 {demand} 单。\n"
        f"相关图表: {chart_paths.get('hourly_demand', '未生成')}"
    )


def answer_weekday_weekend(question: str, df: pd.DataFrame, chart_paths: dict[str, str]) -> str:
    summary = (
        df.groupby("is_weekend")
        .size()
        .reset_index(name="order_count")
        .replace({"is_weekend": {0: "工作日", 1: "周末"}})
    )
    lines = [f"{row['is_weekend']}: {int(row['order_count'])} 单" for _, row in summary.iterrows()]
    return "工作日/周末订单量对比:\n" + "\n".join(lines) + f"\n相关图表: {chart_paths.get('hourly_daytype_demand', '未生成')}"


def answer_top_regions(question: str, df: pd.DataFrame, chart_paths: dict[str, str]) -> str:
    top_regions = df["PULocationID"].value_counts().head(10)
    lines = [f"区域 {int(zone)}: {int(count)} 单" for zone, count in top_regions.items()]
    return "上车热门区域 TOP10:\n" + "\n".join(lines) + f"\n相关图表: {chart_paths.get('top_pickup_zones', '未生成')}"


def answer_fare_relation(question: str, df: pd.DataFrame, chart_paths: dict[str, str]) -> str:
    correlation = df["trip_distance"].corr(df["fare_amount"])
    avg_fare = df["fare_amount"].mean()
    return (
        f"行程距离与车费的皮尔逊相关系数约为 {correlation:.4f}，平均车费约为 {avg_fare:.2f} 美元。\n"
        f"相关图表: {chart_paths.get('distance_fare_scatter', '未生成')}"
    )


def answer_prediction(question: str, df: pd.DataFrame, rf_result: dict, nn_result: dict) -> str:
    hour = _extract_hour(question)
    zone_id = _extract_zone_id(question)
    if hour is None or zone_id is None:
        return "预测问题请包含区域编号和小时，例如“预测区域132在18点的需求”。"

    subset = df[(df["PULocationID"] == zone_id) & (df["pickup_hour"] == hour)]
    if subset.empty:
        estimated_demand = 0
    else:
        estimated_demand = int(
            subset.groupby(["pickup_date", "pickup_hour", "PULocationID"]).size().mean()
        )

    return (
        f"基于历史聚合规律，预测区域 {zone_id} 在 {hour} 点的平均需求约为 {estimated_demand} 单。\n"
        f"随机森林测试集 MAE={rf_result['mae']:.4f}, RMSE={rf_result['rmse']:.4f}; "
        f"神经网络测试集 MAE={nn_result['mae']:.4f}, RMSE={nn_result['rmse']:.4f}。"
    )


def answer_trip_feature(question: str, df: pd.DataFrame, chart_paths: dict[str, str]) -> str:
    if "高峰" in question:
        subset = df[df["is_peak"] == 1]
        label = "高峰期"
    else:
        subset = df
        label = "全部时段"

    avg_duration = subset["trip_duration_min"].mean()
    avg_speed = subset["speed_mph"].mean()
    return (
        f"{label}平均行程时长约为 {avg_duration:.2f} 分钟，平均速度约为 {avg_speed:.2f} mph。\n"
        f"相关图表: {chart_paths.get('congestion_insight', '未生成')}"
    )


def handle_question(question: str, context: dict) -> str:
    df = context["data"]
    chart_paths = context["chart_paths"]
    rf_result = context["rf_result"]
    nn_result = context["nn_result"]

    if ("几点" in question or "点" in question) and ("需求" in question or "订单" in question):
        return answer_hourly_demand(question, df, chart_paths)
    if "工作日" in question or "周末" in question:
        return answer_weekday_weekend(question, df, chart_paths)
    if "热门区域" in question or "top" in question.lower() or "区域排名" in question:
        return answer_top_regions(question, df, chart_paths)
    if "车费" in question and ("距离" in question or "关系" in question):
        return answer_fare_relation(question, df, chart_paths)
    if "预测" in question and "需求" in question:
        return answer_prediction(question, df, rf_result, nn_result)
    if "高峰" in question or "时长" in question or "速度" in question:
        return answer_trip_feature(question, df, chart_paths)
    return "未匹配到规则问题类型。请尝试询问时段需求、工作日/周末对比、热门区域、车费关系、需求预测或高峰特征。"


def run_qa_loop(context: dict) -> None:
    """Run a command-line QA loop based on rule matching."""
    print("问答系统已启动，输入 exit 退出。")
    while True:
        question = input("请输入问题: ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            print("问答系统已退出。")
            break
        if not question:
            print("请输入有效问题。")
            continue
        answer = handle_question(question, context)
        print(answer)
