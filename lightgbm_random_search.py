#!/usr/bin/env python3
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class LightGBMRandomSearchResult:
    """Notebook-friendly result bundle returned by run_lightgbm_random_search()."""

    search: Any
    best_estimator: Any
    best_params: dict[str, Any]
    best_score: float
    best_model_params: dict[str, Any]
    feature_cols: list[str]
    scoring: str


def lazy_import_dependencies() -> dict[str, Any]:
    try:
        import lightgbm as lgb
        import pandas as pd
        from sklearn.model_selection import RandomizedSearchCV
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency. Install at least: pip install lightgbm pandas scikit-learn scipy"
        ) from exc

    return {
        "lgb": lgb,
        "pd": pd,
        "RandomizedSearchCV": RandomizedSearchCV,
    }


def infer_random_search_scoring(task_type: str, scoring: str | None = None) -> str:
    if scoring:
        return scoring
    if task_type == "classification":
        return "roc_auc"
    if task_type == "regression":
        return "neg_mean_squared_error"
    raise ValueError("task_type must be 'classification' or 'regression'.")


def build_backward_selection_random_search_space() -> dict[str, Any]:
    """Match the RandomizedSearchCV search space from the original BFS workflow."""
    from scipy.stats import randint as sp_randint
    from scipy.stats import uniform as sp_uniform

    return {
        "num_leaves": sp_randint(4, 10),
        "max_bin": list(range(200, 2001, 100)),
        "max_depth": sp_randint(2, 4),
        "n_estimators": list(range(20, 301, 20)),
        "scale_pos_weight": [0.8, 1],
        "learning_rate": sp_uniform(0.004, 0.05),
        "min_child_samples": list(range(2000, 14001, 500)),
        "subsample": sp_uniform(0.7, 0.3),
        "colsample_bytree": sp_uniform(0.7, 0.3),
        "reg_alpha": sp_uniform(0.0001, 10),
    }


def run_lightgbm_random_search(
    X_train: Any,
    y_train: Any,
    *,
    task_type: str = "classification",
    random_search_params: dict[str, Any] | None = None,
    base_params: dict[str, Any] | None = None,
    n_iter: int = 20,
    scoring: str | None = None,
    cv: int | Any = 3,
    search_n_jobs: int = 8,
    model_n_jobs: int = 8,
    random_state: int = 2025,
    sample_weight: Any | None = None,
    refit: bool = True,
) -> LightGBMRandomSearchResult:
    """Run one RandomizedSearchCV pass using the original BFS search space."""
    if task_type not in {"classification", "regression"}:
        raise ValueError("task_type must be 'classification' or 'regression'.")

    deps = lazy_import_dependencies()
    lgb = deps["lgb"]
    random_search_cls = deps["RandomizedSearchCV"]

    resolved_base_params = {
        "objective": "binary" if task_type == "classification" else "regression",
        "metric": "auc" if task_type == "classification" else "l2",
        "verbose": -1,
        "n_jobs": model_n_jobs,
    }
    if base_params:
        resolved_base_params.update(base_params)

    estimator_cls = lgb.LGBMClassifier if task_type == "classification" else lgb.LGBMRegressor
    estimator = estimator_cls(**resolved_base_params)
    resolved_scoring = infer_random_search_scoring(task_type, scoring)
    param_distributions = random_search_params or build_backward_selection_random_search_space()

    random_search = random_search_cls(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=resolved_scoring,
        cv=cv,
        n_jobs=search_n_jobs,
        random_state=random_state,
        refit=refit,
    )

    fit_kwargs: dict[str, Any] = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight

    random_search.fit(X_train, y_train, **fit_kwargs)

    feature_cols = list(X_train.columns) if hasattr(X_train, "columns") else []
    best_params = dict(random_search.best_params_)
    best_model_params = {**resolved_base_params, **best_params}

    return LightGBMRandomSearchResult(
        search=random_search,
        best_estimator=random_search.best_estimator_,
        best_params=best_params,
        best_score=float(random_search.best_score_),
        best_model_params=best_model_params,
        feature_cols=feature_cols,
        scoring=resolved_scoring,
    )


def run_lightgbm_random_search_from_df(
    train_df: Any,
    *,
    target_col: str,
    feature_cols: Sequence[str] | None = None,
    drop_cols: Sequence[str] = (),
    sample_weight_col: str = "",
    task_type: str = "classification",
    random_search_params: dict[str, Any] | None = None,
    base_params: dict[str, Any] | None = None,
    n_iter: int = 20,
    scoring: str | None = None,
    cv: int | Any = 3,
    search_n_jobs: int = 8,
    model_n_jobs: int = 8,
    random_state: int = 2025,
    refit: bool = True,
) -> LightGBMRandomSearchResult:
    """DataFrame wrapper for one-off RandomizedSearchCV tuning on selected features."""
    if target_col not in train_df.columns:
        raise KeyError(f"Target column '{target_col}' not found in training dataframe.")

    sample_weight_col = sample_weight_col.strip()
    if feature_cols is None:
        excluded = {target_col, *drop_cols}
        if sample_weight_col:
            excluded.add(sample_weight_col)
        resolved_feature_cols = [column for column in train_df.columns if column not in excluded]
    else:
        resolved_feature_cols = list(feature_cols)

    if not resolved_feature_cols:
        raise ValueError("No feature columns available for random search.")

    missing_feature_cols = [column for column in resolved_feature_cols if column not in train_df.columns]
    if missing_feature_cols:
        raise KeyError(f"Training dataframe is missing feature columns: {missing_feature_cols}")

    sample_weight = train_df[sample_weight_col] if sample_weight_col else None
    return run_lightgbm_random_search(
        train_df.loc[:, resolved_feature_cols],
        train_df[target_col],
        task_type=task_type,
        random_search_params=random_search_params,
        base_params=base_params,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        search_n_jobs=search_n_jobs,
        model_n_jobs=model_n_jobs,
        random_state=random_state,
        sample_weight=sample_weight,
        refit=refit,
    )


__all__ = [
    "LightGBMRandomSearchResult",
    "build_backward_selection_random_search_space",
    "infer_random_search_scoring",
    "run_lightgbm_random_search",
    "run_lightgbm_random_search_from_df",
]
