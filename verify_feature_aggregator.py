from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

DEFAULT_VERIFY_CARD_TYPES = ("debit", "credit", "all")
DEFAULT_VERIFY_SOURCE_GROUPS = ("bank", "cf", "top", "others", "nonloan")
DEFAULT_VERIFY_RESULT_TYPES = ("succ", "fail", "all")
DEFAULT_VERIFY_METRICS = ("ordercnt", "membercnt", "cardcnt")
DEFAULT_VERIFY_DAY_WINDOWS = (3, 7, 14, 21, 31, 60, 90, 180, 360, 720)


def build_verify_feature_name(
    card_type: str,
    source_group: str,
    result_type: str,
    metric: str,
    day_window: int,
) -> str:
    return f"verify_{card_type}_{source_group}_{result_type}_{metric}_day{day_window}"


def list_verify_source_feature_columns(
    *,
    card_types: Sequence[str] = DEFAULT_VERIFY_CARD_TYPES,
    source_groups: Sequence[str] = DEFAULT_VERIFY_SOURCE_GROUPS,
    result_types: Sequence[str] = DEFAULT_VERIFY_RESULT_TYPES,
    metrics: Sequence[str] = DEFAULT_VERIFY_METRICS,
    day_windows: Sequence[int] = DEFAULT_VERIFY_DAY_WINDOWS,
) -> list[str]:
    return [
        build_verify_feature_name(card_type, source_group, result_type, metric, day_window)
        for card_type in card_types
        for source_group in source_groups
        for result_type in result_types
        for metric in metrics
        for day_window in day_windows
    ]


def list_verify_all_feature_columns(
    *,
    card_types: Sequence[str] = DEFAULT_VERIFY_CARD_TYPES,
    result_types: Sequence[str] = DEFAULT_VERIFY_RESULT_TYPES,
    metrics: Sequence[str] = DEFAULT_VERIFY_METRICS,
    day_windows: Sequence[int] = DEFAULT_VERIFY_DAY_WINDOWS,
) -> list[str]:
    return [
        build_verify_feature_name(card_type, "all", result_type, metric, day_window)
        for card_type in card_types
        for result_type in result_types
        for metric in metrics
        for day_window in day_windows
    ]


def add_verify_all_source_group_features(
    df: pd.DataFrame,
    *,
    card_types: Sequence[str] = DEFAULT_VERIFY_CARD_TYPES,
    source_groups: Sequence[str] = DEFAULT_VERIFY_SOURCE_GROUPS,
    result_types: Sequence[str] = DEFAULT_VERIFY_RESULT_TYPES,
    metrics: Sequence[str] = DEFAULT_VERIFY_METRICS,
    day_windows: Sequence[int] = DEFAULT_VERIFY_DAY_WINDOWS,
    fill_value: float | int = 0,
    strict: bool = True,
    copy: bool = True,
) -> pd.DataFrame:
    """Aggregate bank/cf/top/others/nonloan columns into source_group='all'."""
    result_df = df.copy() if copy else df

    required_source_cols = list_verify_source_feature_columns(
        card_types=card_types,
        source_groups=source_groups,
        result_types=result_types,
        metrics=metrics,
        day_windows=day_windows,
    )
    missing_columns = [column for column in required_source_cols if column not in result_df.columns]
    if strict and missing_columns:
        preview = ", ".join(missing_columns[:10])
        suffix = "" if len(missing_columns) <= 10 else ", ..."
        raise KeyError(f"Missing verify source columns: {preview}{suffix}")

    numeric_source_df = (
        result_df.reindex(columns=required_source_cols)
        .apply(pd.to_numeric, errors="coerce")
        .fillna(fill_value)
    )

    for card_type in card_types:
        for result_type in result_types:
            for metric in metrics:
                for day_window in day_windows:
                    target_col = build_verify_feature_name(card_type, "all", result_type, metric, day_window)
                    source_cols = [
                        build_verify_feature_name(card_type, source_group, result_type, metric, day_window)
                        for source_group in source_groups
                    ]
                    result_df[target_col] = numeric_source_df.loc[:, source_cols].sum(axis=1)

    return result_df
