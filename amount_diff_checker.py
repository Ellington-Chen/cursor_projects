from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from tqdm.auto import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    tqdm = None
    TQDM_AVAILABLE = False

DEFAULT_SAMPLE_SEED = 42

RESULT_COLUMNS = [
    "dateBack",
    "card",
    "idcard",
    "source1",
    "source2",
    "amount_json_col",
    "institution1",
    "institution2",
    "value1",
    "value2",
    "diff",
]


def _unique_columns(columns: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(columns))


def _require_columns(df: pd.DataFrame, required_columns: Sequence[str]) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"缺少必要列: {missing_columns}")


def _iter_with_text_progress(
    iterable: Any,
    *,
    total: int | None = None,
    desc: str = "处理中",
) -> Any:
    """无 tqdm 时退化为轻量文本进度输出。"""
    if total == 0:
        return iterable

    progress_step = 1 if total is None else max(1, total // 20)

    def generator() -> Any:
        last_reported = 0
        for index, item in enumerate(iterable, start=1):
            should_report = (
                index == 1
                or total is None
                or index == total
                or (index - last_reported) >= progress_step
            )
            if should_report:
                if total is None:
                    print(f"{desc}: 已处理 {index}")
                else:
                    percent = (index / total) * 100
                    print(f"{desc}: {index}/{total} ({percent:.1f}%)")
                last_reported = index
            yield item

    return generator()


def _iter_with_progress(
    iterable: Any,
    *,
    enabled: bool,
    total: int | None = None,
    desc: str = "处理中",
) -> Any:
    if not enabled:
        return iterable
    if TQDM_AVAILABLE:
        return tqdm(iterable, total=total, desc=desc)
    return _iter_with_text_progress(iterable, total=total, desc=desc)


def _write_unique_cards_csv(
    unique_cards: Sequence[str],
    column_cards: dict[str, list[str]],
    output_path: str | Path | None,
) -> str | None:
    if output_path is None:
        return None

    output_file = Path(output_path).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, list[str]] = {"card": list(unique_cards)}
    for col_name, cards in column_cards.items():
        data[col_name] = list(cards)

    max_len = max((len(v) for v in data.values()), default=0)
    for key in data:
        data[key] += [""] * (max_len - len(data[key]))

    pd.DataFrame(data).to_csv(output_file, index=False, encoding="utf-8-sig")
    return str(output_file)


def parse_amount_json(json_value: Any) -> tuple[dict[str, Any], bool]:
    """解析金额 JSON，并返回是否发生解析异常。"""
    if pd.isna(json_value) or json_value == "":
        return {}, False

    if isinstance(json_value, Mapping):
        return dict(json_value), False

    if not isinstance(json_value, str):
        return {}, True

    try:
        parsed = json.loads(json_value)
    except json.JSONDecodeError:
        return {}, True

    if isinstance(parsed, Mapping):
        return dict(parsed), False

    return {}, True


def sample_data_by_keys(
    df: pd.DataFrame,
    key_cols: Sequence[str],
    sample_rate: float = 0.1,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
) -> pd.DataFrame:
    """按给定键组合抽样，sample_seed 固定时结果可复现。"""
    if not 0 < sample_rate <= 1:
        raise ValueError("sample_rate 必须在 (0, 1] 区间内")

    key_cols = list(key_cols)
    _require_columns(df, key_cols)

    unique_keys = df.loc[:, key_cols].drop_duplicates()
    if unique_keys.empty:
        return df.iloc[0:0].copy()

    sampled_keys = unique_keys.sample(frac=sample_rate, random_state=sample_seed)
    return df.merge(sampled_keys, on=key_cols, how="inner")


def sample_data_by_idcard_dateback(
    df: pd.DataFrame,
    idcard_col: str,
    dateback_col: str,
    sample_rate: float = 0.1,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
) -> pd.DataFrame:
    """兼容旧接口：按 idcard + dateBack 抽样，并显式支持 seed。"""
    return sample_data_by_keys(
        df=df,
        key_cols=[idcard_col, dateback_col],
        sample_rate=sample_rate,
        sample_seed=sample_seed,
    )


def _build_long_amount_frame(
    df: pd.DataFrame,
    amount_json_col: str,
    *,
    group_keys: Sequence[str],
    source_col: str,
    idcard_col: str,
    dateback_col: str,
    card_col: str,
    show_progress: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    parsed_dicts: list[dict[str, Any]] = []
    parse_error_count = 0

    value_iterable = _iter_with_progress(
        df[amount_json_col].tolist(),
        enabled=show_progress,
        total=len(df),
        desc=f"解析 {amount_json_col}",
    )
    for value in value_iterable:
        parsed, had_error = parse_amount_json(value)
        parsed_dicts.append(parsed)
        parse_error_count += int(had_error)

    exploded_pairs = pd.Series(
        [list(amounts.items()) for amounts in parsed_dicts],
        index=df.index,
        name="_amount_pairs",
    ).explode()

    base_columns = _unique_columns([*group_keys, idcard_col, dateback_col, card_col, source_col])
    if exploded_pairs.dropna().empty:
        empty_frame = pd.DataFrame(columns=[*base_columns, "institution", "value", "amount_json_col"])
        return empty_frame, {
            "parsed_row_count": int(len(df)),
            "expanded_row_count": 0,
            "candidate_group_count": 0,
            "parse_error_count": parse_error_count,
        }

    exploded_pairs = exploded_pairs.dropna()
    long_df = df.loc[exploded_pairs.index, base_columns].copy()
    amount_parts = pd.DataFrame(
        exploded_pairs.tolist(),
        index=exploded_pairs.index,
        columns=["institution", "value"],
    )
    long_df = long_df.join(amount_parts)
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df = long_df.dropna(subset=["institution", "value"]).copy()
    long_df["institution"] = long_df["institution"].astype(str)
    long_df["amount_json_col"] = amount_json_col

    dedupe_columns = _unique_columns([*group_keys, source_col, "institution", "value"])
    long_df = long_df.drop_duplicates(subset=dedupe_columns).reset_index(drop=True)

    candidate_mask = long_df.groupby(list(group_keys), sort=False, dropna=False)[source_col].transform("nunique") > 1
    long_df = long_df.loc[candidate_mask].copy()

    candidate_group_count = 0 if long_df.empty else int(long_df.loc[:, list(group_keys)].drop_duplicates().shape[0])
    return long_df.reset_index(drop=True), {
        "parsed_row_count": int(len(df)),
        "expanded_row_count": int(len(long_df)),
        "candidate_group_count": candidate_group_count,
        "parse_error_count": parse_error_count,
    }


def _collect_matches_for_group(
    group: pd.DataFrame,
    *,
    amount_json_col: str,
    idcard_col: str,
    dateback_col: str,
    card_col: str,
    source_col: str,
    diff_threshold: float,
) -> list[dict[str, Any]]:
    if len(group) < 2 or diff_threshold <= 0:
        return []

    ordered = group.sort_values(["value", source_col, "institution"], kind="mergesort").reset_index(drop=True)

    values = ordered["value"].to_numpy(dtype="float64", copy=False)
    sources = ordered[source_col].to_numpy(dtype=object, copy=False)
    institutions = ordered["institution"].to_numpy(dtype=object, copy=False)
    idcards = ordered[idcard_col].to_numpy(dtype=object, copy=False)
    datebacks = ordered[dateback_col].to_numpy(dtype=object, copy=False)
    cards = ordered[card_col].to_numpy(dtype=object, copy=False)

    results: list[dict[str, Any]] = []
    right = 1
    size = len(ordered)

    for left in range(size - 1):
        if right < left + 1:
            right = left + 1

        while right < size and (values[right] - values[left]) < diff_threshold:
            right += 1

        for candidate in range(left + 1, right):
            if sources[left] == sources[candidate]:
                continue
            if institutions[left] == institutions[candidate]:
                continue

            results.append(
                {
                    "dateBack": datebacks[left],
                    "card": cards[left],
                    "idcard": idcards[left],
                    "source1": sources[left],
                    "source2": sources[candidate],
                    "amount_json_col": amount_json_col,
                    "institution1": institutions[left],
                    "institution2": institutions[candidate],
                    "value1": float(values[left]),
                    "value2": float(values[candidate]),
                    "diff": float(values[candidate] - values[left]),
                }
            )

    return results


def check_card_amount_diffs_different_institutions_refactored(
    df: pd.DataFrame,
    *,
    use_sampling: bool = True,
    sample_rate: float = 0.1,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
    idcard_col: str = "idcard",
    dateback_col: str = "dateBack",
    card_col: str = "card",
    source_col: str = "source",
    amount_json_cols: Sequence[str] = ("pre_1_bank_fail_out_max",),
    diff_threshold: float = 1.0,
    group_keys: Sequence[str] | None = None,
    sample_key_cols: Sequence[str] | None = None,
    show_progress: bool = True,
    unique_cards_output_path: str | Path | None = "unique_cards.csv",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    更稳、更快的重构版：
    1. 默认按 idcard + dateBack + card 分组，避免把不同身份证的人混到一起。
    2. 使用长表 + 排序滑窗，只比较差值阈值内的候选，避免 O(source^2 * institution^2) 爆炸。
    3. sample_seed 显式可配，便于复现。
    4. 支持进度可视化，并将 unique cards 单独导出为 CSV。
    """
    amount_json_cols = list(amount_json_cols)
    if not amount_json_cols:
        raise ValueError("amount_json_cols 不能为空")

    group_keys = list(group_keys) if group_keys is not None else [idcard_col, dateback_col, card_col]
    sample_key_cols = list(sample_key_cols) if sample_key_cols is not None else list(group_keys)

    required_columns = _unique_columns(
        [idcard_col, dateback_col, card_col, source_col, *group_keys, *sample_key_cols, *amount_json_cols]
    )
    _require_columns(df, required_columns)

    working_df = df.copy()
    sampled_group_count = None
    sampled_row_count = int(len(working_df))

    if use_sampling:
        working_df = sample_data_by_keys(
            df=working_df,
            key_cols=sample_key_cols,
            sample_rate=sample_rate,
            sample_seed=sample_seed,
        )
        sampled_group_count = int(working_df.loc[:, sample_key_cols].drop_duplicates().shape[0])
        sampled_row_count = int(len(working_df))
    else:
        sampled_group_count = int(working_df.loc[:, sample_key_cols].drop_duplicates().shape[0])

    result_frames: list[pd.DataFrame] = []
    column_stats: dict[str, dict[str, Any]] = {}
    column_cards: dict[str, list[str]] = {}
    total_parse_error_count = 0
    total_candidate_group_count = 0

    amount_col_iterable = _iter_with_progress(
        amount_json_cols,
        enabled=show_progress,
        total=len(amount_json_cols),
        desc="处理金额列",
    )
    for amount_json_col in amount_col_iterable:
        long_df, build_stats = _build_long_amount_frame(
            working_df,
            amount_json_col,
            group_keys=group_keys,
            source_col=source_col,
            idcard_col=idcard_col,
            dateback_col=dateback_col,
            card_col=card_col,
            show_progress=show_progress,
        )
        total_parse_error_count += int(build_stats["parse_error_count"])
        total_candidate_group_count += int(build_stats["candidate_group_count"])

        column_results: list[dict[str, Any]] = []
        grouped = long_df.groupby(group_keys, sort=False, dropna=False)
        group_iterable = _iter_with_progress(
            grouped,
            enabled=show_progress,
            total=build_stats["candidate_group_count"],
            desc=f"匹配 {amount_json_col}",
        )
        for _, group in group_iterable:
            column_results.extend(
                _collect_matches_for_group(
                    group,
                    amount_json_col=amount_json_col,
                    idcard_col=idcard_col,
                    dateback_col=dateback_col,
                    card_col=card_col,
                    source_col=source_col,
                    diff_threshold=diff_threshold,
                )
            )

        column_result_df = pd.DataFrame(column_results, columns=RESULT_COLUMNS)
        result_frames.append(column_result_df)

        cards = [] if column_result_df.empty else sorted(column_result_df["card"].dropna().astype(str).unique().tolist())
        column_cards[amount_json_col] = cards
        column_stats[amount_json_col] = {
            **build_stats,
            "result_count": int(len(column_result_df)),
            "card_count": len(cards),
            "cards": cards,
        }

    if result_frames:
        results_df = pd.concat(result_frames, ignore_index=True)
    else:
        results_df = pd.DataFrame(columns=RESULT_COLUMNS)

    unique_cards = [] if results_df.empty else sorted(results_df["card"].dropna().astype(str).unique().tolist())
    unique_cards_csv_output = _write_unique_cards_csv(unique_cards, column_cards, unique_cards_output_path)
    summary = {
        "input_row_count": int(len(df)),
        "sampled_row_count": sampled_row_count,
        "sampled_group_count": sampled_group_count,
        "sample_rate": sample_rate if use_sampling else 1.0,
        "sample_seed": sample_seed,
        "group_keys": list(group_keys),
        "sample_key_cols": list(sample_key_cols),
        "total_results": int(len(results_df)),
        "total_columns": len(amount_json_cols),
        "total_candidate_group_count": total_candidate_group_count,
        "total_unique_cards": len(unique_cards),
        "unique_cards_output_path": unique_cards_csv_output,
        "parse_error_count": total_parse_error_count,
        "column_stats": column_stats,
    }
    return results_df, summary


def check_card_amount_diffs_different_institutions_fast(
    df: pd.DataFrame,
    force_restart: bool = False,
    use_sampling: bool = True,
    sample_rate: float = 0.1,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
    use_dask: bool = False,
    npartitions: int | None = None,
    idcard_col: str = "idcard",
    dateback_col: str = "dateBack",
    card_col: str = "card",
    source_col: str = "source",
    amount_json_cols: Sequence[str] = ("pre_1_bank_fail_out_max",),
    diff_threshold: float = 1.0,
    group_keys: Sequence[str] | None = None,
    sample_key_cols: Sequence[str] | None = None,
    show_progress: bool = True,
    unique_cards_output_path: str | Path | None = "unique_cards.csv",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    兼容旧函数名的包装器。

    force_restart / use_dask / npartitions 参数保留仅为兼容旧调用方式；
    这个重构版不再使用不安全的 Dask 分区逻辑。
    """
    del force_restart, use_dask, npartitions

    results_df, summary = check_card_amount_diffs_different_institutions_refactored(
        df=df,
        use_sampling=use_sampling,
        sample_rate=sample_rate,
        sample_seed=sample_seed,
        idcard_col=idcard_col,
        dateback_col=dateback_col,
        card_col=card_col,
        source_col=source_col,
        amount_json_cols=amount_json_cols,
        diff_threshold=diff_threshold,
        group_keys=group_keys,
        sample_key_cols=sample_key_cols,
        show_progress=show_progress,
        unique_cards_output_path=unique_cards_output_path,
    )
    return results_df.to_dict("records"), summary


__all__ = [
    "DEFAULT_SAMPLE_SEED",
    "RESULT_COLUMNS",
    "check_card_amount_diffs_different_institutions_fast",
    "check_card_amount_diffs_different_institutions_refactored",
    "parse_amount_json",
    "sample_data_by_idcard_dateback",
    "sample_data_by_keys",
]
