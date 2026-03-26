# cursor_projects

## amount_diff_checker

新增了一个更稳、更快的金额差异检查脚本：`amount_diff_checker.py`。

### 解决的问题

- 默认按 `idcard + dateBack + card` 分组，避免把不同身份证的人混到同一组。
- 抽样支持显式 `sample_seed`，同样的数据和参数下结果可复现。
- 不再使用会漏检的 Dask 分区方案。
- 使用“长表 + 按金额排序后的滑窗比较”来减少无效组合比较。

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
    amount_json_cols=["pre_1_bank_fail_out_max"],
    diff_threshold=1.0,
)
```

### 运行自测

```bash
python -m tests.test_amount_diff_checker
```
