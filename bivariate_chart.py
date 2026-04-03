#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_MISSING_LABEL = "Missing"


@dataclass(slots=True)
class BivariateChartResult:
    """Result bundle returned by build_bivariate_summary()."""

    feature_col: str
    target_col: str
    summary: pd.DataFrame
    interval_rule: str
    requested_bin_count: int
    actual_bin_count: int
    missing_label: str
    bin_edges: list[float]


def _require_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")


def _format_number(value: float, precision: int) -> str:
    if pd.isna(value):
        return "nan"
    text = f"{float(value):.{precision}f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def _build_interval_labels(bin_edges: list[float], *, precision: int) -> list[str]:
    labels: list[str] = []
    for index, (left_edge, right_edge) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        left_bracket = "[" if index == 0 else "("
        labels.append(
            f"{left_bracket}{_format_number(left_edge, precision)}, "
            f"{_format_number(right_edge, precision)}]"
        )
    return labels


def _build_interval_rule(actual_bin_count: int, missing_label: str) -> str:
    if actual_bin_count <= 0:
        return f"{missing_label} is shown as a standalone bin."
    return (
        f"{missing_label} is shown as a standalone bin. "
        "Non-missing values use approximately equal-frequency bins. "
        "The first bin is left-closed and right-closed: [a, b]. "
        "Later bins are left-open and right-closed: (b, c]."
    )


def build_bivariate_summary(
    df: pd.DataFrame,
    *,
    feature_col: str,
    target_col: str,
    n_bins: int = 6,
    missing_label: str = DEFAULT_MISSING_LABEL,
    precision: int = 6,
) -> BivariateChartResult:
    """Build bivariate summary data for a numeric feature against a numeric target."""
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1.")

    _require_columns(df, [feature_col, target_col])

    summary_columns = [
        "bin",
        "count",
        "count_ratio",
        "mean_label",
        "valid_target_count",
        "is_missing",
        "left_endpoint",
        "right_endpoint",
        "left_closed",
        "right_closed",
        "min_value",
        "max_value",
    ]

    if df.empty:
        return BivariateChartResult(
            feature_col=feature_col,
            target_col=target_col,
            summary=pd.DataFrame(columns=summary_columns),
            interval_rule=_build_interval_rule(actual_bin_count=0, missing_label=missing_label),
            requested_bin_count=n_bins,
            actual_bin_count=0,
            missing_label=missing_label,
            bin_edges=[],
        )

    numeric_feature = pd.to_numeric(df[feature_col], errors="coerce")
    numeric_target = pd.to_numeric(df[target_col], errors="coerce")

    if df[target_col].notna().any() and numeric_target.notna().sum() == 0:
        raise TypeError(
            "target_col must be numeric or convertible to numeric so mean_label can be calculated."
        )

    missing_mask = numeric_feature.isna()
    non_missing_feature = numeric_feature.loc[~missing_mask]

    bin_labels = pd.Series(index=df.index, dtype=object)
    metadata_rows: list[dict[str, Any]] = []
    category_order: list[str] = []
    interval_labels: list[str] = []
    bin_edges: list[float] = []
    actual_bin_count = 0

    if missing_mask.any():
        category_order.append(missing_label)
        metadata_rows.append(
            {
                "bin": missing_label,
                "is_missing": True,
                "left_endpoint": float("nan"),
                "right_endpoint": float("nan"),
                "left_closed": False,
                "right_closed": False,
            }
        )
        bin_labels.loc[missing_mask] = missing_label

    if not non_missing_feature.empty:
        unique_value_count = int(non_missing_feature.nunique(dropna=True))
        if unique_value_count == 1:
            only_value = float(non_missing_feature.iloc[0])
            interval_labels = [
                f"[{_format_number(only_value, precision)}, {_format_number(only_value, precision)}]"
            ]
            bin_edges = [only_value, only_value]
        else:
            requested_bin_count = min(n_bins, unique_value_count)
            quantile_codes, raw_edges = pd.qcut(
                non_missing_feature,
                q=requested_bin_count,
                labels=False,
                retbins=True,
                duplicates="drop",
            )
            bin_edges = [float(edge) for edge in raw_edges]
            interval_labels = _build_interval_labels(bin_edges, precision=precision)
            quantile_codes = pd.Series(quantile_codes, index=non_missing_feature.index)
            bin_labels.loc[non_missing_feature.index] = quantile_codes.map(
                lambda code: interval_labels[int(code)]
            )

        if unique_value_count == 1:
            bin_labels.loc[non_missing_feature.index] = interval_labels[0]

        actual_bin_count = len(interval_labels)
        category_order.extend(interval_labels)
        for index, label in enumerate(interval_labels):
            left_edge = bin_edges[index]
            right_edge = bin_edges[index + 1]
            metadata_rows.append(
                {
                    "bin": label,
                    "is_missing": False,
                    "left_endpoint": left_edge,
                    "right_endpoint": right_edge,
                    "left_closed": index == 0,
                    "right_closed": True,
                }
            )

    grouped = (
        pd.DataFrame(
            {
                "bin": pd.Categorical(bin_labels, categories=category_order, ordered=True),
                "_feature": numeric_feature,
                "_target": numeric_target,
            }
        )
        .groupby("bin", observed=False, dropna=False)
        .agg(
            count=("_feature", "size"),
            mean_label=("_target", "mean"),
            valid_target_count=("_target", "count"),
            min_value=("_feature", "min"),
            max_value=("_feature", "max"),
        )
        .reset_index()
    )

    summary = pd.DataFrame(metadata_rows).merge(grouped, on="bin", how="left")
    summary["count"] = summary["count"].fillna(0).astype(int)
    summary["valid_target_count"] = summary["valid_target_count"].fillna(0).astype(int)
    summary["count_ratio"] = summary["count"] / len(df)
    summary = summary.loc[summary["count"] > 0, summary_columns].reset_index(drop=True)

    return BivariateChartResult(
        feature_col=feature_col,
        target_col=target_col,
        summary=summary,
        interval_rule=_build_interval_rule(actual_bin_count=actual_bin_count, missing_label=missing_label),
        requested_bin_count=n_bins,
        actual_bin_count=actual_bin_count,
        missing_label=missing_label,
        bin_edges=bin_edges,
    )


def plot_bivariate_chart(
    result: BivariateChartResult,
    *,
    title: str | None = None,
    figsize: tuple[int, int] = (12, 6),
    count_color: str = "tab:blue",
    label_color: str = "red",
    rotate_xticks: int = 45,
    save_path: str | Path | None = None,
) -> tuple[Any, tuple[Any, Any]]:
    """Plot the bivariate chart from a BivariateChartResult."""
    if result.summary.empty:
        raise ValueError("No data is available to plot.")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - import path depends on environment
        raise ImportError(
            "plot_bivariate_chart requires matplotlib. Install it with: pip install matplotlib"
        ) from exc

    fig, ax_count = plt.subplots(figsize=figsize)
    ax_label = ax_count.twinx()

    summary = result.summary.copy()
    x_positions = list(range(len(summary)))

    ax_count.bar(x_positions, summary["count"], color=count_color, alpha=0.65, label="Count")
    ax_label.plot(
        x_positions,
        summary["mean_label"],
        color=label_color,
        marker="o",
        linewidth=2.0,
        label="Mean label",
    )

    ax_count.set_xticks(x_positions)
    ax_count.set_xticklabels(summary["bin"].tolist(), rotation=rotate_xticks, ha="right")
    ax_count.set_xlabel("Bin")
    ax_count.set_ylabel("Count", color=count_color)
    ax_label.set_ylabel("Mean label", color=label_color)
    ax_count.set_title(title or f"{result.feature_col} vs {result.target_col}")

    handles_1, labels_1 = ax_count.get_legend_handles_labels()
    handles_2, labels_2 = ax_label.get_legend_handles_labels()
    ax_count.legend(handles_1 + handles_2, labels_1 + labels_2, loc="upper right")

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.text(0.01, 0.01, result.interval_rule, ha="left", va="bottom", fontsize=9)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, (ax_count, ax_label)


def build_and_plot_bivariate_chart(
    df: pd.DataFrame,
    *,
    feature_col: str,
    target_col: str,
    n_bins: int = 6,
    missing_label: str = DEFAULT_MISSING_LABEL,
    precision: int = 6,
    title: str | None = None,
    figsize: tuple[int, int] = (12, 6),
    count_color: str = "tab:blue",
    label_color: str = "red",
    rotate_xticks: int = 45,
    save_path: str | Path | None = None,
) -> tuple[BivariateChartResult, tuple[Any, tuple[Any, Any]]]:
    """Convenience wrapper that builds the summary first and then plots it."""
    result = build_bivariate_summary(
        df,
        feature_col=feature_col,
        target_col=target_col,
        n_bins=n_bins,
        missing_label=missing_label,
        precision=precision,
    )
    figure_and_axes = plot_bivariate_chart(
        result,
        title=title,
        figsize=figsize,
        count_color=count_color,
        label_color=label_color,
        rotate_xticks=rotate_xticks,
        save_path=save_path,
    )
    return result, figure_and_axes


__all__ = [
    "BivariateChartResult",
    "DEFAULT_MISSING_LABEL",
    "build_and_plot_bivariate_chart",
    "build_bivariate_summary",
    "plot_bivariate_chart",
]
