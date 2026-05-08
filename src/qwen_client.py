from __future__ import annotations

import os
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency at runtime
    OpenAI = None  # type: ignore[assignment]


QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_DEFAULT_MODEL = "qwen-plus"


def is_qwen_available() -> bool:
    """Return whether the local environment is ready for Qwen API calls."""
    return OpenAI is not None and bool(os.getenv("DASHSCOPE_API_KEY"))


def build_qwen_system_prompt(context_summary: dict[str, Any]) -> str:
    """Create a constrained system prompt for fallback explanatory answers."""
    return f"""
你是一个“纽约出租车出行数据问答助手”，负责回答用户关于本地分析结果的问题。

你的回答必须遵守以下规则：
1. 只能基于本地项目已经计算出的统计信息、图表结论和模型指标进行解释。
2. 不允许编造不存在的数据、区域名称、时间段结论或模型效果。
3. 如果用户问题超出了本地结果范围，要明确说明“本地规则系统暂未直接支持该问题”，再给出尽可能有帮助的解释。
4. 回答风格要简洁、中文、解释性强，适合作业展示。
5. 如果提到模型，请优先引用以下真实指标：
   - 随机森林 MAE: {context_summary["rf_mae"]:.4f}
   - 随机森林 RMSE: {context_summary["rf_rmse"]:.4f}
   - 神经网络 MAE: {context_summary["nn_mae"]:.4f}
   - 神经网络 RMSE: {context_summary["nn_rmse"]:.4f}
6. 如果用户的问题和图表相关，可以建议查看对应图表路径，但不要捏造不存在的路径。

本地上下文摘要：
- 数据总量（清洗后）: {context_summary["row_count"]}
- 图表数量: {context_summary["chart_count"]}
- 热门上车区域前3: {context_summary["top_pickup_preview"]}
- 平均车费: {context_summary["avg_fare"]:.2f}
- 平均行程距离: {context_summary["avg_distance"]:.2f}
- 高峰定义: 工作日常见通勤时段 7-9 点、17-19 点
""".strip()


def ask_qwen(question: str, context_summary: dict[str, Any]) -> str:
    """Call Qwen via DashScope's OpenAI-compatible endpoint."""
    if OpenAI is None:
        return "当前环境未安装 openai 依赖，无法调用 Qwen 兜底问答。"

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return "未检测到 DASHSCOPE_API_KEY，当前仅能使用规则问答。"

    client = OpenAI(api_key=api_key, base_url=QWEN_BASE_URL)
    model_name = os.getenv("QWEN_MODEL", QWEN_DEFAULT_MODEL)

    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": build_qwen_system_prompt(context_summary)},
            {"role": "user", "content": question},
        ],
        temperature=0.3,
    )

    message = completion.choices[0].message.content
    return message.strip() if message else "Qwen 未返回有效内容。"
