# NYC Taxi 出行数据问答系统

## 1. 项目简介

本项目基于纽约市 TLC 公布的 `yellow_tripdata_2023-01` 黄出租车行程数据，构建了一个完整的 Python 出行数据问答系统，覆盖课程作业要求的四个模块：

- M1 数据处理
- M2 分析可视化
- M3 预测模型
- M4 命令行问答接口

系统能够完成从原始数据加载、清洗、特征工程、统计分析、图表输出，到出行需求预测与自然语言问答的全流程处理。项目主入口为 `main.py`，满足“一键运行”的作业要求。

## 2. 数据集说明

数据来源：

- NYC Taxi & Limousine Commission Trip Record Data
- 黄出租车公开数据：2023 年 1 月

使用文件：

- `data/raw/yellow_tripdata_2023-01.parquet`

本项目主要保留以下字段用于分析与建模：

- `tpep_pickup_datetime`
- `tpep_dropoff_datetime`
- `passenger_count`
- `trip_distance`
- `PULocationID`
- `DOLocationID`
- `fare_amount`
- `total_amount`
- `tip_amount`
- `payment_type`

## 3. 项目结构

```text
AI_Prog_HW/
├─ data/
│  ├─ raw/
│  │  └─ yellow_tripdata_2023-01.parquet
│  └─ processed/
├─ outputs/
│  ├─ figures/
│  ├─ models/
│  └─ reports/
├─ src/
│  ├─ __init__.py
│  ├─ analysis.py
│  ├─ data_loader.py
│  ├─ feature_engineering.py
│  ├─ model_nn.py
│  ├─ model_rf.py
│  ├─ preprocess.py
│  ├─ qa_system.py
│  └─ utils.py
├─ main.py
├─ requirements.txt
├─ 人机协作报告.md
└─ README.md
```

## 4. 环境要求

- Python 3.10+
- 建议使用 Conda 虚拟环境

推荐环境创建方式：

```powershell
conda create -n work python=3.10 -y
conda activate work
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 5. 代码克隆与完整使用流程

### 5.1 克隆项目

如果你的代码已经上传到 GitHub 或 Gitee，可以使用下面的命令克隆仓库：

```powershell
git clone https://github.com/你的用户名/你的仓库名.git
cd 你的仓库名
```

如果你使用的是 Gitee，也可以替换为：

```powershell
git clone https://gitee.com/你的用户名/你的仓库名.git
cd 你的仓库名
```

### 5.2 创建并激活虚拟环境

推荐使用 Conda：

```powershell
conda create -n work python=3.10 -y
conda activate work
```

如果你已经提前创建过环境，只需要激活即可：

```powershell
conda activate work
```

### 5.3 安装依赖

进入项目根目录后执行：

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

建议始终使用 `python -m pip`，这样可以避免包装错到其他 Python 环境中。

### 5.4 准备数据集

请先从 NYC TLC 官方网站下载黄出租车 2023 年 1 月数据：

- `yellow_tripdata_2023-01.parquet`

下载完成后，将文件放到以下路径：

```text
data/raw/yellow_tripdata_2023-01.parquet
```

目录结构示例：

```text
data/
├─ raw/
│  └─ yellow_tripdata_2023-01.parquet
└─ processed/
```

### 5.5 运行主程序

在激活虚拟环境、安装依赖、放置好数据文件后，执行：

```powershell
python main.py
```

程序会自动完成：

1. 数据加载
2. 数据质量报告生成
3. 数据清洗
4. 特征工程
5. 分析可视化
6. 随机森林建模
7. 神经网络建模
8. 模型对比结果输出
9. 命令行问答系统启动

### 5.6 问答系统使用示例

当程序运行到最后，会进入命令行问答模式：

```text
问答系统已启动，输入 exit 退出。
请输入问题:
```

你可以输入如下问题：

```text
18点需求多少？
工作日和周末订单量有什么区别？
热门区域有哪些？
车费和距离有什么关系？
预测区域132在18点的需求
高峰期平均行程时长是多少？
随机森林和神经网络哪个更好？
```

退出方式：

```text
exit
```

### 5.7 可选：启用 Qwen 增强问答

如果你希望启用 Qwen 作为规则问答的兜底解释层，需要先准备 DashScope API Key。

在 PowerShell 中设置：

```powershell
$env:DASHSCOPE_API_KEY="你的真实API_KEY"
$env:QWEN_MODEL="qwen-plus"
```

然后重新运行：

```powershell
python main.py
```

如果启用成功，问答系统会提示当前已启用 Qwen 兜底问答。

### 5.8 输出文件说明

程序运行完成后，会自动生成以下内容：

- 清洗后数据：`data/processed/`
- 图表文件：`outputs/figures/`
- 报告文件：`outputs/reports/`

其中包括：

- `quality_report.csv`
- `cleaning_summary.csv`
- `random_forest_metrics.json`
- `random_forest_feature_importance.csv`
- `neural_network_metrics.json`
- `model_comparison.md`
- `nn_loss_curve.png`

### 5.9 一次性复现流程汇总

如果你是第一次在新电脑上运行本项目，可以直接按下面顺序操作：

```powershell
git clone https://github.com/你的用户名/你的仓库名.git
cd 你的仓库名
conda create -n work python=3.10 -y
conda activate work
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

## 6. 依赖说明

项目依赖如下：

```txt
pandas==2.2.3
numpy==1.26.4
pyarrow==16.1.0
matplotlib==3.9.2
seaborn==0.13.2
scikit-learn==1.5.2
torch==2.4.1
tqdm==4.66.5
jupyter==1.1.1
openpyxl==3.1.5
```

说明：

- `pyarrow` 用于读取 parquet 数据
- `scikit-learn` 用于随机森林建模和评价指标
- `torch` 用于构建神经网络预测模型
- `matplotlib` 与 `seaborn` 用于生成图表

## 7. 运行方式

在激活虚拟环境后，直接运行：

```powershell
python main.py
```

程序将自动执行以下流程：

1. 加载原始数据
2. 生成数据质量报告
3. 执行数据清洗
4. 进行特征工程
5. 生成分析图表
6. 训练随机森林模型
7. 训练神经网络模型
8. 输出模型评价结果
9. 进入命令行问答系统

## 8. 各模块说明

### M1 数据处理

实现内容：

- 加载 parquet 数据
- 生成数据质量报告
- 统计每列缺失率
- 统计异常值数量
- 统计重复记录
- 清洗无效和异常记录
- 提取时间特征与衍生特征

主要清洗策略：

- 删除关键字段缺失记录
- 删除重复记录
- 删除下车时间早于上车时间的异常记录
- 删除距离、车费、总费用非正值记录
- 保留合理乘客数范围
- 删除极端异常时长与极端费用样本

生成文件：

- `outputs/reports/quality_report.csv`
- `outputs/reports/cleaning_summary.csv`
- `data/processed/cleaned_tripdata_2023-01.parquet`
- `data/processed/featured_tripdata_2023-01.parquet`

### M2 分析可视化

实现的分析内容：

1. 出行需求时间规律分析
2. 区域热度分析
3. 车费影响因素分析
4. 自选分析：拥堵与行程效率分析

生成图表包括：

- `hourly_demand.png`
- `hourly_weekday_weekend_demand.png`
- `top10_pickup_zones.png`
- `top10_dropoff_zones.png`
- `pickup_zone_hour_heatmap.png`
- `distance_fare_scatter.png`
- `fare_by_time_period_boxplot.png`
- `fare_by_passenger_count.png`
- `congestion_speed_duration.png`

所有图表自动保存到：

- `outputs/figures/`

### M3 预测模型

预测目标：

- 预测“某区域某时段”的出行需求量

建模方式：

- 按 `pickup_date + pickup_hour + PULocationID` 聚合样本
- 构造 `demand_count` 作为预测标签

使用特征：

- `pickup_hour`
- `PULocationID`
- `avg_distance`
- `avg_fare`
- `avg_duration`
- `avg_speed`
- `pickup_weekday`
- `is_weekend`
- `is_peak`
- `prev_hour_demand`

模型包括：

- 随机森林回归 `RandomForestRegressor`
- PyTorch 多层感知机 `MLP`

评价指标：

- MAE
- RMSE

生成文件：

- `outputs/reports/random_forest_metrics.json`
- `outputs/reports/random_forest_feature_importance.csv`
- `outputs/reports/neural_network_metrics.json`
- `outputs/reports/model_comparison.md`
- `outputs/figures/nn_loss_curve.png`

### M4 问答接口

实现形式：

- 基于 `.py` 文件的命令行问答循环

主要思路：

- 接收用户自然语言输入
- 进行关键词匹配和正则提取
- 自动判断问题类型
- 调用前面模块的分析或预测结果
- 返回数字结论与图表路径

当前支持的问题类型包括：

1. 时段需求查询
2. 工作日与周末对比
3. 热门区域查询
4. 车费关系分析
5. 区域时段需求预测
6. 高峰时段特征查询
7. 模型效果对比问答

示例问题：

```text
18点需求多少？
工作日和周末订单量有什么区别？
热门区域有哪些？
车费和距离有什么关系？
预测区域132在18点的需求
高峰期平均行程时长是多少？
随机森林和神经网络哪个更好？
```

### Qwen 增强问答

为了提升系统完整度，项目额外接入了 Qwen 作为兜底问答层。

设计方式：

- 优先使用本地规则系统回答结构化问题
- 当规则系统未命中时，调用 Qwen 给出解释性回复
- Qwen 回答受到 System Prompt 约束，只能基于本地统计结果、图表与模型指标进行解释

对应实现文件：

- `src/qwen_client.py`
- `src/qa_system.py`

### 启用 Qwen 的方法

先安装依赖：

```powershell
python -m pip install -r requirements.txt
```

然后配置 DashScope API Key：

```powershell
$env:DASHSCOPE_API_KEY="你的API_KEY"
```

可选配置模型名：

```powershell
$env:QWEN_MODEL="qwen-plus"
```

之后重新运行：

```powershell
python main.py
```

如果环境变量配置成功，问答系统启动时会提示：

```text
当前已启用 Qwen 兜底问答
```

如果没有配置 API Key，系统仍然可以正常运行，只是只使用规则问答，不影响基础得分。

## 9. 关键特征工程说明

基础时间特征：

- `pickup_hour`
- `pickup_weekday`
- `is_weekend`
- `is_peak`

自定义衍生特征：

- `trip_duration_min`
- `time_period`
- `speed_mph`
- `fare_per_mile`

这些特征既用于分析可视化，也用于后续需求预测。

## 10. 输出结果说明

### 报告类输出

保存在：

- `outputs/reports/`

内容包括：

- 数据质量报告
- 清洗摘要
- 随机森林指标
- 神经网络指标
- 模型对比说明

### 图表类输出

保存在：

- `outputs/figures/`

内容包括：

- 需求规律图
- 区域热度图
- 费用分析图
- 神经网络 loss 曲线
- 自选分析图

## 11. 课程提交建议

提交到 GitHub/Gitee 前，建议确保仓库中包含：

- `main.py`
- `requirements.txt`
- `src/` 源代码目录
- `data/` 目录
- `outputs/` 目录
- `人机协作报告.md`
- `README.md`
- `main.py` 完整运行截图

注意：

- 如果原始数据文件过大，不适合直接上传 GitHub，可以在仓库说明中写明下载链接与放置路径
- 如果课程要求必须包含 `data/` 目录，可以保留目录结构，并在 README 中注明原始文件名

## 12. 已知注意事项

- 本项目默认原始数据文件为 `data/raw/yellow_tripdata_2023-01.parquet`
- 若本地运行时出现 `torch` 导入失败，通常是因为没有激活正确的虚拟环境
- Windows 终端中建议始终使用：

```powershell
python -m pip install -r requirements.txt
```

而不是直接使用 `pip install -r requirements.txt`

## 13. 后续可扩展方向

- 接入区域名称映射表，将 `PULocationID` 和 `DOLocationID` 显示为真实区域名称
- 使用 Streamlit 或 Gradio 构建图形化可视化界面
- 已接入 Qwen，可继续扩展 DeepSeek / GLM 等多模型兜底能力
- 增加更多时序特征，提高需求预测精度

## 14. 作者说明

本项目用于 AI 编程课程作业，重点展示数据处理、分析可视化、预测建模与规则问答系统的完整实现流程，同时保留了人机协作开发过程中的反思与记录。
