#!/usr/bin/env python3
from __future__ import annotations

import gc
import os
import time
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV

try:
    from tqdm.auto import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    tqdm = None
    TQDM_AVAILABLE = False


@dataclass
class ResumeAfterFirstRoundResult:
    """Result bundle for the post-first-round LightGBM workflow."""

    model: Any
    best_params: dict[str, Any]
    model_params: dict[str, Any]
    current_features: list[str]
    feature_importance_df: pd.DataFrame
    metricsTrain: dict[str, float]
    metricsTest: dict[str, float]
    metricsOOT: dict[str, float]
    random_search_best_score: float
    random_search: Any


class _RandomSearchProgressReporter:
    """Notebook / terminal friendly progress reporter for RandomizedSearchCV."""

    def __init__(
        self,
        *,
        description: str,
        total_fits: int,
        total_candidates: int,
        cv_splits: int,
        enabled: bool,
    ) -> None:
        self.description = description
        self.total_fits = max(0, int(total_fits))
        self.total_candidates = max(0, int(total_candidates))
        self.cv_splits = max(1, int(cv_splits))
        self.enabled = enabled and self.total_fits > 0
        self.completed_fits = 0
        self.start_time = time.time()
        self._bar: Any | None = None
        self._last_reported = 0
        self._text_step = max(1, self.total_fits // 10) if self.total_fits > 0 else 1

        if not self.enabled:
            return
        if TQDM_AVAILABLE:
            self._bar = tqdm(total=self.total_fits, desc=description, leave=True)
        else:
            print(
                f"{description}: starting {self.total_fits} CV fit(s) across "
                f"{self.total_candidates} candidate(s) (cv={self.cv_splits})"
            )

    def update(self, count: int) -> None:
        if not self.enabled or count <= 0:
            return

        self.completed_fits = min(self.total_fits, self.completed_fits + int(count))
        completed_candidates = min(self.total_candidates, self.completed_fits // self.cv_splits)

        if self._bar is not None:
            increment = self.completed_fits - self._bar.n
            if increment > 0:
                self._bar.update(increment)
            self._bar.set_postfix(
                {"candidates": f"{completed_candidates}/{self.total_candidates}"},
                refresh=False,
            )
            return

        should_report = (
            self.completed_fits == self.total_fits
            or self.completed_fits == 1
            or (self.completed_fits - self._last_reported) >= self._text_step
        )
        if should_report:
            elapsed = time.time() - self.start_time
            print(
                f"{self.description}: {self.completed_fits}/{self.total_fits} CV fit(s), "
                f"candidates~{completed_candidates}/{self.total_candidates}, elapsed={elapsed:.1f}s"
            )
            self._last_reported = self.completed_fits

    def close(self, *, best_score: float | None = None) -> None:
        if not self.enabled:
            return

        completed_candidates = min(self.total_candidates, self.completed_fits // self.cv_splits)
        best_text = "n/a" if best_score is None else f"{best_score:.6f}"

        if self._bar is not None:
            self._bar.set_postfix(
                {
                    "candidates": f"{completed_candidates}/{self.total_candidates}",
                    "best": best_text,
                },
                refresh=False,
            )
            self._bar.close()
            return

        elapsed = time.time() - self.start_time
        print(f"{self.description}: done, best_score={best_text}, elapsed={elapsed:.1f}s")


def _resolve_cv_split_count(cv: int | Any, X_train: Any, y_train: Any, task_type: str) -> int:
    if isinstance(cv, int):
        return cv

    try:
        from sklearn.model_selection import check_cv

        cv_splitter = check_cv(cv, y=y_train, classifier=(task_type == "classification"))
        return int(cv_splitter.get_n_splits(X_train, y_train))
    except Exception:
        return 1


@contextmanager
def _joblib_progress(reporter: _RandomSearchProgressReporter) -> Any:
    if not reporter.enabled:
        yield
        return

    import joblib

    previous_callback = joblib.parallel.BatchCompletionCallBack

    class _PatchedBatchCompletionCallBack(previous_callback):
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            reporter.update(getattr(self, "batch_size", 1))
            return super().__call__(*args, **kwargs)

    joblib.parallel.BatchCompletionCallBack = _PatchedBatchCompletionCallBack
    try:
        yield
    finally:
        joblib.parallel.BatchCompletionCallBack = previous_callback


def calculate_ks(y_true: Any, y_pred: Any) -> float:
    """Calculate KS statistic."""
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    y_pred_pos = y_pred_array[y_true_array == 1]
    y_pred_neg = y_pred_array[y_true_array == 0]
    return float(ks_2samp(y_pred_pos, y_pred_neg).statistic)


def calculate_top_decile_lift(y_true: Any, y_pred: Any, top_percent: float = 0.1) -> float:
    """Calculate lift in top predictions."""
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    n_top = max(1, int(len(y_pred_array) * top_percent))
    top_indices = np.argsort(y_pred_array)[-n_top:]
    top_actual = y_true_array[top_indices]
    overall_rate = float(np.mean(y_true_array))
    top_rate = float(np.mean(top_actual))
    return top_rate / overall_rate if overall_rate > 0 else 0.0


def get_backward_selection_random_search_space() -> dict[str, Any]:
    """Keep the search range aligned with the original workflow."""
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


def _get_lgb_train_flags() -> tuple[bool, bool]:
    import lightgbm as lgb

    try:
        lgb_version = lgb.__version__
        version_parts = lgb_version.split(".")
        major_version = int(version_parts[0])
        minor_version = int(version_parts[1]) if len(version_parts) > 1 else 0

        if major_version > 4 or (major_version == 4 and minor_version >= 6):
            return True, True
        return False, False
    except Exception:
        return True, True


def run_backward_selection_after_first_round(
    X_train: pd.DataFrame,
    y_train: Any,
    X_val: pd.DataFrame,
    y_val: Any,
    X_oot: pd.DataFrame,
    y_oot: Any,
    *,
    selected_features: Sequence[str],
    task_type: str = "classification",
    eval_metric: Any | None = None,
    random_search_params: dict[str, Any] | None = None,
    n_random_search_iter: int = 20,
    use_ks: bool = True,
    use_lift: bool = True,
    save_model_path: str | None = None,
    weight_column: str | None = None,
    random_search_cv: int | Any = 3,
    random_state: int = 2025,
    random_search_n_jobs: int = 8,
    model_n_jobs: int = 8,
    show_progress: bool = True,
) -> ResumeAfterFirstRoundResult:
    """
    Resume the original workflow after the first fixed-params round.

    Treat `selected_features` as the feature list already retained after the first
    round removed unimportant variables, then start directly from RandomizedSearchCV.
    """
    import lightgbm as lgb

    if task_type not in {"classification", "regression"}:
        raise ValueError("task_type must be 'classification' or 'regression'.")

    current_features = list(selected_features)
    if not current_features:
        raise ValueError("selected_features cannot be empty.")

    for frame_name, frame in (("X_train", X_train), ("X_val", X_val), ("X_oot", X_oot)):
        missing_cols = [column for column in current_features if column not in frame.columns]
        if missing_cols:
            raise KeyError(f"{frame_name} is missing selected_features columns: {missing_cols}")

    if weight_column is not None:
        if weight_column not in X_train.columns:
            raise KeyError(f"weight_column '{weight_column}' not found in X_train.")

    if random_search_params is None:
        random_search_params = get_backward_selection_random_search_space()

    base_params = {
        "objective": "binary" if task_type == "classification" else "regression",
        "metric": "auc" if task_type == "classification" else "l2",
        "verbose": -1,
        "n_jobs": model_n_jobs,
    }

    if eval_metric is None:
        eval_metric = roc_auc_score if task_type == "classification" else mean_squared_error

    if task_type == "classification":
        lgb_model = lgb.LGBMClassifier(**base_params)
        scoring = "roc_auc"
    else:
        lgb_model = lgb.LGBMRegressor(**base_params)
        scoring = "neg_mean_squared_error"

    random_search = RandomizedSearchCV(
        estimator=lgb_model,
        param_distributions=random_search_params,
        n_iter=n_random_search_iter,
        scoring=scoring,
        cv=random_search_cv,
        n_jobs=random_search_n_jobs,
        random_state=random_state,
        refit=True,
    )

    fit_kwargs: dict[str, Any] = {}
    if weight_column is not None:
        fit_kwargs["sample_weight"] = X_train[weight_column].values
        print(f"使用权重列: {weight_column}")

    cv_splits = _resolve_cv_split_count(random_search_cv, X_train[current_features], y_train, task_type)
    progress_reporter = _RandomSearchProgressReporter(
        description="random_search_after_first_round",
        total_fits=max(0, int(n_random_search_iter)) * max(1, cv_splits),
        total_candidates=max(0, int(n_random_search_iter)),
        cv_splits=cv_splits,
        enabled=show_progress,
    )

    try:
        with _joblib_progress(progress_reporter):
            random_search.fit(X_train[current_features], y_train, **fit_kwargs)
    except Exception:
        progress_reporter.close(best_score=None)
        raise

    progress_reporter.close(best_score=float(random_search.best_score_))
    best_params = dict(random_search.best_params_)
    model_params = {**base_params, **best_params}

    X_train_subset = X_train[current_features].astype("float32")
    X_val_subset = X_val[current_features].astype("float32")

    train_kwargs = {
        "data": X_train_subset,
        "label": y_train,
        "categorical_feature": None,
        "free_raw_data": False,
        "params": {"max_bin": model_params.get("max_bin", 255)},
    }

    if weight_column is not None:
        train_kwargs["weight"] = X_train[weight_column].values

    train_data = lgb.Dataset(**train_kwargs)
    valid_data = lgb.Dataset(
        X_val_subset,
        label=y_val,
        reference=train_data,
        free_raw_data=False,
    )

    gc.collect()
    lgbm_use_callbacks, lgbm_feature_importance_compat = _get_lgb_train_flags()

    if lgbm_use_callbacks:
        callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=False)]
        model_params_copy = model_params.copy()
        model_params_copy["verbose"] = -1
        model = lgb.train(
            model_params_copy,
            train_data,
            num_boost_round=100,
            valid_sets=[valid_data],
            callbacks=callbacks,
        )
    else:
        model = lgb.train(
            model_params,
            train_data,
            num_boost_round=100,
            valid_sets=[valid_data],
            early_stopping_rounds=20,
            verbose_eval=False,
        )

    predtrain = model.predict(X_train[current_features])
    predtest = model.predict(X_val[current_features])
    predoot = model.predict(X_oot[current_features])

    if task_type == "classification":
        metricsTrain = {"auc": float(eval_metric(y_train, predtrain))}
        metricsTest = {"auc": float(eval_metric(y_val, predtest))}
        metricsOOT = {"auc": float(eval_metric(y_oot, predoot))}
    else:
        metricsTrain = {"mse": float(eval_metric(y_train, predtrain))}
        metricsTest = {"mse": float(eval_metric(y_val, predtest))}
        metricsOOT = {"mse": float(eval_metric(y_oot, predoot))}

    if task_type == "classification":
        if use_ks:
            metricsTrain["ks"] = round(calculate_ks(y_train, predtrain), 3)
            metricsTest["ks"] = round(calculate_ks(y_val, predtest), 3)
            metricsOOT["ks"] = round(calculate_ks(y_oot, predoot), 3)

        if use_lift:
            metricsTrain["lift"] = round(calculate_top_decile_lift(y_train, predtrain), 3)
            metricsTest["lift"] = round(calculate_top_decile_lift(y_val, predtest), 3)
            metricsOOT["lift"] = round(calculate_top_decile_lift(y_oot, predoot), 3)

    if lgbm_feature_importance_compat:
        try:
            importance = model.feature_importance(importance_type="gain")
        except Exception:
            try:
                importance = model._booster.feature_importance(importance_type="gain")
            except Exception:
                importance = [1.0] * len(current_features)
    else:
        try:
            importance = model.feature_importance(importance_type="gain")
        except Exception:
            importance = [1.0] * len(current_features)

    feature_importance_df = pd.DataFrame(
        {"feature": current_features, "gain": importance}
    ).sort_values("gain", ascending=False)

    print(metricsTrain)
    print(metricsTest)
    print(metricsOOT)

    del train_data, valid_data
    gc.collect()

    if save_model_path is not None:
        os.makedirs(save_model_path, exist_ok=True)
        model_file_path = os.path.join(save_model_path, f"model{len(current_features)}.txt")
        model.save_model(model_file_path)
        print(f"模型已保存至 {model_file_path}")

    return ResumeAfterFirstRoundResult(
        model=model,
        best_params=best_params,
        model_params=model_params,
        current_features=current_features,
        feature_importance_df=feature_importance_df,
        metricsTrain=metricsTrain,
        metricsTest=metricsTest,
        metricsOOT=metricsOOT,
        random_search_best_score=float(random_search.best_score_),
        random_search=random_search,
    )


__all__ = [
    "ResumeAfterFirstRoundResult",
    "calculate_ks",
    "calculate_top_decile_lift",
    "get_backward_selection_random_search_space",
    "run_backward_selection_after_first_round",
]
