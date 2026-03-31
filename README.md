# cursor_projects

## amount_diff_checker

新增了一个更稳、更快的金额差异检查脚本：`amount_diff_checker.py`。

### 解决的问题

- 默认按 `idcard + dateBack + card` 分组，避免把不同身份证的人混到同一组。
- 抽样支持显式 `sample_seed`，同样的数据和参数下结果可复现。
- 不再使用会漏检的 Dask 分区方案。
- 使用“长表 + 按金额排序后的滑窗比较”来减少无效组合比较。
- 支持进度可视化；如果安装了 `tqdm` 会显示进度条，否则退化为轻量文本进度。
- `summary` 不再返回 `unique_cards`，唯一卡号会单独导出为 CSV。

### 主要入口

- `check_card_amount_diffs_different_institutions_refactored`
- `check_card_amount_diffs_different_institutions_fast`  
  兼容旧函数名，内部会走重构版实现。

### 简单示例

```python
import pandas as pd

from amount_diff_checker import check_card_amount_diffs_different_institutions_fast

df = pd.DataFrame(
    [
        {
            "idcard": "id-1",
            "dateBack": "2026-01-01",
            "card": "card-1",
            "source": "src-a",
            "pre_1_bank_fail_out_max": '{"inst-b": 100.0}',
        },
        {
            "idcard": "id-1",
            "dateBack": "2026-01-01",
            "card": "card-1",
            "source": "src-b",
            "pre_1_bank_fail_out_max": '{"inst-a": 100.4}',
        },
    ]
)

results, summary = check_card_amount_diffs_different_institutions_fast(
    df,
    use_sampling=True,
    sample_rate=1.0,
    sample_seed=7,
    show_progress=True,
    unique_cards_csv_path="unique_cards.csv",
    amount_json_cols=["pre_1_bank_fail_out_max"],
    diff_threshold=1.0,
)
```

`summary` 中会保留：

- `total_unique_cards`
- `unique_cards_csv_path`

但不会再内嵌 `unique_cards` 列表。

### 运行自测

```bash
python3 -m tests.test_amount_diff_checker
```

## lightgbm_optuna_tuner

新增了一个专门给 **LightGBM + Optuna** 用的调参脚本：`lightgbm_optuna_tuner.py`。

### 适用场景

- 只想调 LightGBM，不需要完整 AutoML
- 数据量较大，例如你这种 `train + test + oot` 合计约 93 万行
- 希望优先优化调参速度，而不是直接做很重的 KFold / 全量搜索

### 主要速度优化点

- **两阶段搜索**
  - 第一阶段：小样本粗搜
  - 第二阶段：较大样本精搜
- **只用单验证集，不做交叉验证**
- **early stopping + Optuna pruning**
- **自动限制每个 trial 的线程数**
  - 避免机器核数不多时，多个 trial 抢资源导致反而更慢
- **最后才用全量 train + valid 重训**
  - 调参阶段不在全量 93 万数据上硬跑全部 trial
- **固定部分 Dataset 级参数**
  - 例如 `max_bin`、`feature_pre_filter`
  - 这样可以安全复用 LightGBM Dataset，避免每个 trial 重建分桶结构

### 输入方式

支持两种方式：

1. 一个总表，靠 `split` 列切分  
2. 分别传 `train / valid / oot` 三个文件

支持文件格式：

- CSV
- Parquet
- Feather

### 推荐做法

如果你的数据已经有：

- `train`
- `test` 或 `valid`
- `oot`

建议把：

- `train` 当作训练集
- `test/valid` 当作调参验证集
- `oot` 只用于最终评估

### 示例 1：一个文件 + split 列

假设你的总表叫 `data/all_data.parquet`，其中：

- `split=train` 表示训练集
- `split=test` 表示验证集
- `split=oot` 表示 oot

```bash
python3 lightgbm_optuna_tuner.py \
  --data-path data/all_data.parquet \
  --split-col split \
  --train-values train \
  --valid-values test \
  --oot-values oot \
  --target-col target \
  --task binary \
  --drop-cols user_id,apply_time \
  --categorical-cols city,channel,product_code \
  --fast-phase-trials 12 \
  --main-phase-trials 24 \
  --fast-phase-train-rows 120000 \
  --fast-phase-valid-rows 60000 \
  --max-tune-train-rows 250000 \
  --max-tune-valid-rows 120000 \
  --trial-parallelism 1 \
  --num-threads 4 \
  --output-dir artifacts/lgbm_optuna_run
```

### 示例 2：分别传 train / valid / oot

```bash
python3 lightgbm_optuna_tuner.py \
  --train-path data/train.parquet \
  --valid-path data/test.parquet \
  --oot-path data/oot.parquet \
  --target-col target \
  --task binary \
  --drop-cols user_id,apply_time \
  --categorical-cols city,channel,product_code \
  --fast-phase-trials 12 \
  --main-phase-trials 24 \
  --max-tune-train-rows 250000 \
  --max-tune-valid-rows 120000 \
  --trial-parallelism 1 \
  --num-threads 4
```

### 输出产物

默认输出到：

```text
artifacts/lgbm_optuna/
```

主要文件包括：

- `lightgbm_model.txt`：最终模型
- `best_params.json`：最佳参数
- `summary.json`：行数、特征数、阶段结果、OOT 评估等摘要
- `phase1_trials.csv`：第一阶段 trial 明细
- `phase2_trials.csv`：第二阶段 trial 明细

### 针对 93 万行数据的参数建议

如果你机器资源一般，建议先从下面这组开始：

- `--fast-phase-trials 8~12`
- `--main-phase-trials 20~30`
- `--fast-phase-train-rows 100000~150000`
- `--max-tune-train-rows 200000~300000`
- `--trial-parallelism 1`
- `--num-threads` 设成机器可用核数

如果是 4 核机器，通常更建议：

- `trial_parallelism=1`
- `num_threads=4`

而不是：

- `trial_parallelism=2`
- `num_threads=2`

因为 LightGBM 单 trial 本身就能很好吃掉 CPU，多 trial 并行在小机器上容易互相争资源。

### 依赖安装

脚本依赖：

```bash
pip install lightgbm optuna pandas scikit-learn pyarrow
```

如果你只读 CSV，可以不装 `pyarrow`。

### 自测

```bash
python3 -m unittest tests.test_lightgbm_optuna_tuner
```
