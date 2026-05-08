from __future__ import annotations

from src.analysis import run_all_analyses
from src.data_loader import load_trip_data
from src.feature_engineering import add_time_features
from src.model_nn import train_neural_network
from src.model_rf import train_random_forest
from src.preprocess import build_quality_report, clean_trip_data
from src.qa_system import run_qa_loop
from src.utils import RAW_DIR, ensure_directories, print_step


def main() -> None:
    ensure_directories()

    file_path = RAW_DIR / "yellow_tripdata_2023-01.parquet"

    print_step("加载数据")
    df = load_trip_data(file_path)
    print(f"原始数据形状: {df.shape}")

    print_step("数据质量报告")
    quality_report = build_quality_report(df)
    print(f"质量报告行数: {len(quality_report)}")

    print_step("清洗与特征工程")
    cleaned_df = clean_trip_data(df)
    feature_df = add_time_features(cleaned_df)
    print(f"处理后数据形状: {feature_df.shape}")

    print_step("分析可视化")
    chart_paths = run_all_analyses(feature_df)
    print(f"已生成图表数量: {len(chart_paths)}")

    print_step("随机森林建模")
    rf_result = train_random_forest(feature_df)
    print(f"随机森林结果键: {list(rf_result.keys())}")

    print_step("神经网络建模")
    nn_result = train_neural_network(feature_df)
    print(f"神经网络结果键: {list(nn_result.keys())}")

    print_step("问答系统")
    run_qa_loop(
        {
            "data": feature_df,
            "quality_report": quality_report,
            "chart_paths": chart_paths,
            "rf_result": rf_result,
            "nn_result": nn_result,
        }
    )


if __name__ == "__main__":
    main()
