from __future__ import annotations

from src.analysis import run_all_analyses
from src.data_loader import get_required_columns, load_trip_data
from src.feature_engineering import add_time_features
from src.model_nn import train_neural_network
from src.model_rf import train_random_forest
from src.preprocess import build_quality_report, clean_trip_data
from src.qa_system import run_qa_loop
from src.utils import RAW_DIR, REPORTS_DIR, ensure_directories, print_step


def save_model_comparison(rf_result: dict, nn_result: dict) -> None:
    """Save a simple text comparison between random forest and neural network."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    lines = [
        "# 模型对比分析",
        "",
        f"- 随机森林 MAE: {rf_result['mae']:.4f}",
        f"- 随机森林 RMSE: {rf_result['rmse']:.4f}",
        f"- 神经网络 MAE: {nn_result['mae']:.4f}",
        f"- 神经网络 RMSE: {nn_result['rmse']:.4f}",
        "",
        "## 对比结论",
        "",
        "1. 随机森林对表格型聚合特征通常更稳健，对异常值和特征尺度不敏感，训练过程也更容易调试。",
        "2. 神经网络能够学习更复杂的非线性关系，但更依赖特征标准化、训练轮数和超参数设置。",
        "3. 在本任务中，若两者性能接近，优先说明随机森林在结构化交通数据上的工程稳定性更强。",
        "4. 若神经网络指标更优，则可以强调它对区域、时间和历史需求交互关系的表达能力更强。",
    ]

    (REPORTS_DIR / "model_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_directories()

    file_path = RAW_DIR / "yellow_tripdata_2023-01.parquet"

    print_step("加载数据")
    df = load_trip_data(file_path, columns=get_required_columns())
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
    print(f"随机森林 MAE: {rf_result['mae']:.4f}, RMSE: {rf_result['rmse']:.4f}")

    print_step("神经网络建模")
    nn_result = train_neural_network(feature_df)
    print(f"神经网络 MAE: {nn_result['mae']:.4f}, RMSE: {nn_result['rmse']:.4f}")

    save_model_comparison(rf_result, nn_result)

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
