from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

DEFAULT_VERIFY_CARD_TYPES = ("debit", "credit", "all")
DEFAULT_VERIFY_SOURCE_GROUPS = ("bank", "cf", "top", "others", "nonloan")
DEFAULT_VERIFY_RESULT_TYPES = ("succ", "fail", "all")
DEFAULT_VERIFY_METRICS = ("ordercnt", "membercnt", "cardcnt")
DEFAULT_VERIFY_MONTH_FEATURE_METRICS = (*DEFAULT_VERIFY_METRICS, "all")
ALL_VERIFY_DAY_WINDOWS = (3, 7, 14, 21, 31, 60, 90, 180, 360, 720)
DEFAULT_VERIFY_DAY_WINDOWS = (31, 60, 90, 180, 360, 720)
DROPPED_VERIFY_DAY_WINDOWS = (3, 7, 14, 21)
VERIFY_DAY_TO_MONTH = {
    31: 1,
    60: 2,
    90: 3,
    180: 6,
    360: 12,
    720: 24,
}
DEFAULT_LAST_BIND_MONTHS_COLUMN = "verify_all_all_all_last_bind_months"
DEFAULT_FIRST_BIND_MONTHS_COLUMN = "verify_all_all_all_first_bind_months"


def build_verify_feature_name(
    card_type: str,
    source_group: str,
    result_type: str,
    metric: str,
    day_window: int,
) -> str:
    return f"verify_{card_type}_{source_group}_{result_type}_{metric}_day{day_window}"


def build_verify_last_bind_months_feature_name(
    card_type: str,
    result_type: str,
    metric: str,
    *,
    source_group: str = "all",
) -> str:
    return f"verify_{card_type}_{source_group}_{result_type}_{metric}_last_bind_months"


def build_verify_first_bind_months_feature_name(
    card_type: str,
    result_type: str,
    metric: str,
    *,
    source_group: str = "all",
) -> str:
    return f"verify_{card_type}_{source_group}_{result_type}_{metric}_first_bind_months"


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


def list_verify_last_bind_months_feature_columns(
    *,
    card_types: Sequence[str] = DEFAULT_VERIFY_CARD_TYPES,
    result_types: Sequence[str] = DEFAULT_VERIFY_RESULT_TYPES,
    metrics: Sequence[str] = DEFAULT_VERIFY_MONTH_FEATURE_METRICS,
    source_group: str = "all",
) -> list[str]:
    return [
        build_verify_last_bind_months_feature_name(
            card_type,
            result_type,
            metric,
            source_group=source_group,
        )
        for card_type in card_types
        for result_type in result_types
        for metric in metrics
    ]


def list_verify_first_bind_months_feature_columns(
    *,
    card_types: Sequence[str] = DEFAULT_VERIFY_CARD_TYPES,
    result_types: Sequence[str] = DEFAULT_VERIFY_RESULT_TYPES,
    metrics: Sequence[str] = DEFAULT_VERIFY_MONTH_FEATURE_METRICS,
    source_group: str = "all",
) -> list[str]:
    return [
        build_verify_first_bind_months_feature_name(
            card_type,
            result_type,
            metric,
            source_group=source_group,
        )
        for card_type in card_types
        for result_type in result_types
        for metric in metrics
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
    """
    Aggregate bank/cf/top/others/nonloan columns into source_group='all'.
    """
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

    aggregated_columns: dict[str, pd.Series] = {}
    for card_type in card_types:
        for result_type in result_types:
            for metric in metrics:
                for day_window in day_windows:
                    target_col = build_verify_feature_name(card_type, "all", result_type, metric, day_window)
                    source_cols = [
                        build_verify_feature_name(card_type, source_group, result_type, metric, day_window)
                        for source_group in source_groups
                    ]
                    aggregated_columns[target_col] = numeric_source_df.loc[:, source_cols].sum(axis=1)

    aggregated_df = pd.DataFrame(aggregated_columns, index=result_df.index)
    result_df.loc[:, aggregated_df.columns] = aggregated_df
    return result_df


def _compute_bind_months_from_windows(
    numeric_df: pd.DataFrame,
    *,
    columns_by_day_window: dict[int, list[str]],
    day_to_month: dict[int, int],
    prefer: str,
) -> pd.Series:
    if prefer not in {"nearest", "furthest"}:
        raise ValueError("prefer must be either 'nearest' or 'furthest'")

    bind_months = pd.Series(np.nan, index=numeric_df.index, dtype="float64")
    month_items = sorted(
        day_to_month.items(),
        key=lambda item: item[1],
        reverse=prefer == "furthest",
    )
    for day_window, month in month_items:
        has_activity = (numeric_df.loc[:, columns_by_day_window[day_window]] > 0).any(axis=1)
        update_mask = bind_months.isna() & has_activity
        bind_months.loc[update_mask] = float(month)
    return bind_months


def _build_metric_columns_by_day_window(
    *,
    card_type: str,
    source_group: str,
    result_type: str,
    metric: str,
    raw_metrics: Sequence[str],
    day_to_month: dict[int, int],
) -> dict[int, list[str]]:
    metric_group = raw_metrics if metric == "all" else (metric,)
    return {
        day_window: [
            build_verify_feature_name(card_type, source_group, result_type, current_metric, day_window)
            for current_metric in metric_group
        ]
        for day_window in day_to_month
    }


def add_verify_last_bind_months(
    df: pd.DataFrame,
    *,
    card_types: Sequence[str] = DEFAULT_VERIFY_CARD_TYPES,
    result_types: Sequence[str] = DEFAULT_VERIFY_RESULT_TYPES,
    raw_metrics: Sequence[str] = DEFAULT_VERIFY_METRICS,
    feature_metrics: Sequence[str] = DEFAULT_VERIFY_MONTH_FEATURE_METRICS,
    day_to_month: dict[int, int] = VERIFY_DAY_TO_MONTH,
    source_group: str = "all",
    column_name: str = DEFAULT_LAST_BIND_MONTHS_COLUMN,
    first_bind_column_name: str = DEFAULT_FIRST_BIND_MONTHS_COLUMN,
    add_legacy_overall_features: bool = True,
    add_expanded_features: bool = True,
    strict: bool = True,
    copy: bool = True,
) -> pd.DataFrame:
    """
    Add the overall and expanded nearest/furthest bind-month bucket features.
    """
    result_df = df.copy() if copy else df

    required_cols: set[str] = set()
    if add_legacy_overall_features:
        required_cols.update(
            build_verify_feature_name("all", source_group, "all", metric, day_window)
            for day_window in day_to_month
            for metric in raw_metrics
        )
    if add_expanded_features:
        required_cols.update(
            build_verify_feature_name(card_type, source_group, result_type, metric, day_window)
            for card_type in card_types
            for result_type in result_types
            for metric in raw_metrics
            for day_window in day_to_month
        )

    required_col_list = sorted(required_cols)
    missing_columns = [column for column in required_col_list if column not in result_df.columns]
    if strict and missing_columns:
        preview = ", ".join(missing_columns[:10])
        suffix = "" if len(missing_columns) <= 10 else ", ..."
        raise KeyError(f"Missing verify all-group columns: {preview}{suffix}")

    numeric_df = (
        result_df.reindex(columns=required_col_list)
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    if add_legacy_overall_features:
        overall_columns_by_day_window = _build_metric_columns_by_day_window(
            card_type="all",
            source_group=source_group,
            result_type="all",
            metric="all",
            raw_metrics=raw_metrics,
            day_to_month=day_to_month,
        )
        result_df[column_name] = _compute_bind_months_from_windows(
            numeric_df,
            columns_by_day_window=overall_columns_by_day_window,
            day_to_month=day_to_month,
            prefer="nearest",
        )
        result_df[first_bind_column_name] = _compute_bind_months_from_windows(
            numeric_df,
            columns_by_day_window=overall_columns_by_day_window,
            day_to_month=day_to_month,
            prefer="furthest",
        )

    if add_expanded_features:
        for card_type in card_types:
            for result_type in result_types:
                for metric in feature_metrics:
                    detailed_columns_by_day_window = _build_metric_columns_by_day_window(
                        card_type=card_type,
                        source_group=source_group,
                        result_type=result_type,
                        metric=metric,
                        raw_metrics=raw_metrics,
                        day_to_month=day_to_month,
                    )
                    result_df[
                        build_verify_last_bind_months_feature_name(
                            card_type,
                            result_type,
                            metric,
                            source_group=source_group,
                        )
                    ] = _compute_bind_months_from_windows(
                        numeric_df,
                        columns_by_day_window=detailed_columns_by_day_window,
                        day_to_month=day_to_month,
                        prefer="nearest",
                    )
                    result_df[
                        build_verify_first_bind_months_feature_name(
                            card_type,
                            result_type,
                            metric,
                            source_group=source_group,
                        )
                    ] = _compute_bind_months_from_windows(
                        numeric_df,
                        columns_by_day_window=detailed_columns_by_day_window,
                        day_to_month=day_to_month,
                        prefer="furthest",
                    )

    return result_df


def transform_verify_features(
    df: pd.DataFrame,
    *,
    card_types: Sequence[str] = DEFAULT_VERIFY_CARD_TYPES,
    source_groups: Sequence[str] = DEFAULT_VERIFY_SOURCE_GROUPS,
    result_types: Sequence[str] = DEFAULT_VERIFY_RESULT_TYPES,
    metrics: Sequence[str] = DEFAULT_VERIFY_METRICS,
    keep_day_windows: Sequence[int] = DEFAULT_VERIFY_DAY_WINDOWS,
    drop_day_windows: Sequence[int] = DROPPED_VERIFY_DAY_WINDOWS,
    fill_value: float | int = 0,
    strict: bool = True,
    add_last_bind_months_feature: bool = True,
    last_bind_months_column: str = DEFAULT_LAST_BIND_MONTHS_COLUMN,
    copy: bool = True,
) -> pd.DataFrame:
    """
    1. Aggregate source_group='all' for kept day windows.
    2. Drop 3/7/14/21 windows and all source-group-level verify columns.
    3. Optionally add the overall and expanded nearest/furthest bind-month bucket features.
    """
    result_df = add_verify_all_source_group_features(
        df,
        card_types=card_types,
        source_groups=source_groups,
        result_types=result_types,
        metrics=metrics,
        day_windows=keep_day_windows,
        fill_value=fill_value,
        strict=strict,
        copy=copy,
    )

    source_level_columns = list_verify_source_feature_columns(
        card_types=card_types,
        source_groups=source_groups,
        result_types=result_types,
        metrics=metrics,
        day_windows=tuple(dict.fromkeys((*keep_day_windows, *drop_day_windows))),
    )
    dropped_window_columns = list_verify_all_feature_columns(
        card_types=card_types,
        result_types=result_types,
        metrics=metrics,
        day_windows=drop_day_windows,
    )
    drop_columns = [column for column in (*source_level_columns, *dropped_window_columns) if column in result_df.columns]
    if drop_columns:
        if copy:
            result_df = result_df.drop(columns=drop_columns)
        else:
            result_df.drop(columns=drop_columns, inplace=True)

    if add_last_bind_months_feature:
        day_to_month = {day_window: VERIFY_DAY_TO_MONTH[day_window] for day_window in keep_day_windows}
        result_df = add_verify_last_bind_months(
            result_df,
            raw_metrics=metrics,
            day_to_month=day_to_month,
            column_name=last_bind_months_column,
            strict=strict,
            copy=False,
        )

    return result_df
