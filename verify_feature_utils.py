from __future__ import annotations

import numpy as np
import pandas as pd

VERIFY_IMAGE_CARD_TYPE = "all"
VERIFY_IMAGE_RESULT_TYPE = "all"
VERIFY_IMAGE_SOURCE_GROUPS = ("bank", "cf", "top", "others", "nonloan")
VERIFY_IMAGE_METRICS = ("ordercnt", "membercnt")
VERIFY_IMAGE_DAY_WINDOWS = (31, 60, 90, 180, 360, 720)
VERIFY_DAY_TO_MONTH = {
    31: 1,
    60: 2,
    90: 3,
    180: 6,
    360: 12,
    720: 24,
}


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


VERIFY_IMAGE_FIRST_BIND_MONTHS_COLUMN = build_verify_first_bind_months_feature_name(
    VERIFY_IMAGE_CARD_TYPE,
    VERIFY_IMAGE_RESULT_TYPE,
    "membercnt",
)
VERIFY_IMAGE_LAST_BIND_MONTHS_COLUMN = build_verify_last_bind_months_feature_name(
    VERIFY_IMAGE_CARD_TYPE,
    VERIFY_IMAGE_RESULT_TYPE,
    "membercnt",
)


def list_verify_image_source_columns() -> list[str]:
    return [
        build_verify_feature_name(
            VERIFY_IMAGE_CARD_TYPE,
            source_group,
            VERIFY_IMAGE_RESULT_TYPE,
            metric,
            day_window,
        )
        for source_group in VERIFY_IMAGE_SOURCE_GROUPS
        for metric in VERIFY_IMAGE_METRICS
        for day_window in VERIFY_IMAGE_DAY_WINDOWS
    ]


def list_verify_image_all_columns() -> list[str]:
    return [
        build_verify_feature_name(
            VERIFY_IMAGE_CARD_TYPE,
            "all",
            VERIFY_IMAGE_RESULT_TYPE,
            metric,
            day_window,
        )
        for metric in VERIFY_IMAGE_METRICS
        for day_window in VERIFY_IMAGE_DAY_WINDOWS
    ]


def list_verify_image_feature_columns() -> list[str]:
    return [
        VERIFY_IMAGE_FIRST_BIND_MONTHS_COLUMN,
        VERIFY_IMAGE_LAST_BIND_MONTHS_COLUMN,
        *list_verify_image_all_columns(),
        *list_verify_image_source_columns(),
    ]


def _compute_bind_months_from_windows(
    numeric_all_df: pd.DataFrame,
    *,
    prefer: str,
) -> pd.Series:
    if prefer not in {"nearest", "furthest"}:
        raise ValueError("prefer must be either 'nearest' or 'furthest'")

    bind_months = pd.Series(np.nan, index=numeric_all_df.index, dtype="float64")
    ordered_day_windows = sorted(
        VERIFY_IMAGE_DAY_WINDOWS,
        key=lambda day_window: VERIFY_DAY_TO_MONTH[day_window],
        reverse=prefer == "furthest",
    )
    for day_window in ordered_day_windows:
        column = build_verify_feature_name(
            VERIFY_IMAGE_CARD_TYPE,
            "all",
            VERIFY_IMAGE_RESULT_TYPE,
            "membercnt",
            day_window,
        )
        has_activity = numeric_all_df[column] > 0
        update_mask = bind_months.isna() & has_activity
        bind_months.loc[update_mask] = float(VERIFY_DAY_TO_MONTH[day_window])
    return bind_months


def transform_verify_features(
    df: pd.DataFrame,
    *,
    fill_value: float | int = 0,
    strict: bool = True,
    keep_only_target_columns: bool = True,
    copy: bool = True,
) -> pd.DataFrame:
    """
    Generate only the verify fields needed by the plotting sheet.

    Required input columns are the source-level day features for
    bank/cf/top/others/nonloan and the metrics ordercnt/membercnt.
    Output columns are:
    1. verify_all_all_all_membercnt_first_bind_months
    2. verify_all_all_all_membercnt_last_bind_months
    3. verify_all_all_all_{ordercnt,membercnt}_day{31,60,90,180,360,720}
    4. verify_all_{bank,cf,top,others,nonloan}_all_{ordercnt,membercnt}_day{31,60,90,180,360,720}
    """
    result_df = df.copy() if copy else df

    source_columns = list_verify_image_source_columns()
    missing_columns = [column for column in source_columns if column not in result_df.columns]
    if strict and missing_columns:
        preview = ", ".join(missing_columns[:10])
        suffix = "" if len(missing_columns) <= 10 else ", ..."
        raise KeyError(f"Missing verify source columns: {preview}{suffix}")

    numeric_source_df = (
        result_df.reindex(columns=source_columns)
        .apply(pd.to_numeric, errors="coerce")
        .fillna(fill_value)
    )
    result_df.loc[:, source_columns] = numeric_source_df

    aggregated_columns: dict[str, pd.Series] = {}
    for metric in VERIFY_IMAGE_METRICS:
        for day_window in VERIFY_IMAGE_DAY_WINDOWS:
            target_col = build_verify_feature_name(
                VERIFY_IMAGE_CARD_TYPE,
                "all",
                VERIFY_IMAGE_RESULT_TYPE,
                metric,
                day_window,
            )
            source_metric_columns = [
                build_verify_feature_name(
                    VERIFY_IMAGE_CARD_TYPE,
                    source_group,
                    VERIFY_IMAGE_RESULT_TYPE,
                    metric,
                    day_window,
                )
                for source_group in VERIFY_IMAGE_SOURCE_GROUPS
            ]
            aggregated_columns[target_col] = numeric_source_df.loc[:, source_metric_columns].sum(axis=1)

    aggregated_df = pd.DataFrame(aggregated_columns, index=result_df.index)
    result_df.loc[:, aggregated_df.columns] = aggregated_df
    result_df[VERIFY_IMAGE_FIRST_BIND_MONTHS_COLUMN] = _compute_bind_months_from_windows(
        aggregated_df,
        prefer="furthest",
    )
    result_df[VERIFY_IMAGE_LAST_BIND_MONTHS_COLUMN] = _compute_bind_months_from_windows(
        aggregated_df,
        prefer="nearest",
    )

    if keep_only_target_columns:
        return result_df.reindex(columns=list_verify_image_feature_columns())
    return result_df


def build_verify_image_variables(
    df: pd.DataFrame,
    *,
    fill_value: float | int = 0,
    strict: bool = True,
    copy: bool = True,
) -> pd.DataFrame:
    return transform_verify_features(
        df,
        fill_value=fill_value,
        strict=strict,
        keep_only_target_columns=True,
        copy=copy,
    )
