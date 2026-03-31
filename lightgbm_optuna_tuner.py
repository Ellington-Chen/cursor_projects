#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from tqdm.auto import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    tqdm = None
    TQDM_AVAILABLE = False


DEFAULT_OUTPUT_DIR = Path("artifacts/lgbm_optuna")
DEFAULT_TRAIN_VALUES = "train"
DEFAULT_VALID_VALUES = "valid,test,dev"
DEFAULT_OOT_VALUES = "oot"


@dataclass(slots=True)
class LightGBMOptunaResult:
    """Notebook-friendly result bundle returned by run_lightgbm_optuna()."""

    model: Any
    best_params: dict[str, Any]
    summary: dict[str, Any]
    feature_cols: list[str]
    categorical_cols: list[str]
    label_mapping: dict[str, int]
    phase_summaries: list[dict[str, Any]]
    studies: dict[str, Any | None]
    artifact_paths: dict[str, str | None]
    best_trial_number: int
    best_tuning_value: float
    calibrated_best_iteration: int
    calibration_score: float
    oot_score: float | None


class _StudyProgressReporter:
    """Notebook / terminal friendly progress reporter for Optuna phases."""

    def __init__(
        self,
        *,
        phase_name: str,
        total_trials: int,
        direction: str,
        enabled: bool,
    ) -> None:
        self.phase_name = phase_name
        self.total_trials = total_trials
        self.direction = direction
        self.enabled = enabled and total_trials > 0
        self.best_value: float | None = None
        self.start_time = time.time()
        self._bar: Any | None = None
        self._last_reported_count = 0
        self._text_step = max(1, total_trials // 10) if total_trials > 0 else 1
        self._text_started = False

        if not self.enabled:
            return
        if TQDM_AVAILABLE:
            self._bar = tqdm(total=total_trials, desc=phase_name, leave=True)

    def __call__(self, study: Any, trial: Any) -> None:
        del trial
        if not self.enabled:
            return

        completed_count = len(study.trials)
        self.best_value = self._extract_best_value(study)

        if self._bar is not None:
            increment = completed_count - self._bar.n
            if increment > 0:
                self._bar.update(increment)
            postfix = self._build_postfix()
            if postfix:
                self._bar.set_postfix(postfix, refresh=False)
            return

        should_report = (
            completed_count == self.total_trials
            or completed_count == 1
            or (completed_count - self._last_reported_count) >= self._text_step
        )
        if should_report:
            best_text = "n/a" if self.best_value is None else f"{self.best_value:.6f}"
            elapsed = time.time() - self.start_time
            if not self._text_started:
                self._text_started = True
            print(
                f"{self.phase_name}: {completed_count}/{self.total_trials} trials, "
                f"best={best_text}, elapsed={elapsed:.1f}s"
            )
            self._last_reported_count = completed_count

    def close(self) -> None:
        if self._bar is not None:
            postfix = self._build_postfix()
            if postfix:
                self._bar.set_postfix(postfix, refresh=False)
            self._bar.close()

    def _build_postfix(self) -> dict[str, str]:
        if self.best_value is None:
            return {}
        key = "best(max)" if self.direction == "maximize" else "best(min)"
        return {key: f"{self.best_value:.6f}"}

    def _extract_best_value(self, study: Any) -> float | None:
        completed_trials = [trial for trial in study.trials if getattr(trial, "value", None) is not None]
        if not completed_trials:
            return None
        values = [float(trial.value) for trial in completed_trials]
        return max(values) if self.direction == "maximize" else min(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optuna-based LightGBM tuner optimized for large tabular datasets. "
            "The default workflow uses two-stage tuning on sampled train/valid "
            "subsets, then calibrates best_iteration on full train/valid before "
            "training the final model on train+valid."
        )
    )
    parser.add_argument("--data-path", type=Path, help="Single CSV/Parquet/Feather file containing all splits.")
    parser.add_argument("--split-col", default="split", help="Split column name when --data-path is used.")
    parser.add_argument(
        "--train-values",
        default=DEFAULT_TRAIN_VALUES,
        help=f"Comma separated split values treated as train. Default: {DEFAULT_TRAIN_VALUES}",
    )
    parser.add_argument(
        "--valid-values",
        default=DEFAULT_VALID_VALUES,
        help=f"Comma separated split values treated as valid/test. Default: {DEFAULT_VALID_VALUES}",
    )
    parser.add_argument(
        "--oot-values",
        default=DEFAULT_OOT_VALUES,
        help=f"Comma separated split values treated as oot. Default: {DEFAULT_OOT_VALUES}",
    )
    parser.add_argument("--train-path", type=Path, help="Standalone train file.")
    parser.add_argument("--valid-path", type=Path, help="Standalone valid/test file.")
    parser.add_argument("--oot-path", type=Path, help="Standalone oot file.")
    parser.add_argument("--target-col", required=True, help="Target column name.")
    parser.add_argument(
        "--task",
        choices=("binary", "multiclass", "regression"),
        default="binary",
        help="Learning task type.",
    )
    parser.add_argument("--metric", help="LightGBM metric. If omitted, a task-specific default is used.")
    parser.add_argument(
        "--drop-cols",
        default="",
        help="Comma separated columns to exclude from features, e.g. id,apply_time.",
    )
    parser.add_argument(
        "--categorical-cols",
        default="",
        help="Comma separated categorical columns. Object/string columns are also auto-detected.",
    )
    parser.add_argument(
        "--sample-weight-col",
        default="",
        help="Optional sample weight column. The column is excluded from features.",
    )
    parser.add_argument(
        "--positive-label",
        default="",
        help="Positive label for binary tasks when target is not already 0/1.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for model and tuning artifacts. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument(
        "--num-boost-round",
        type=int,
        default=3000,
        help="Upper bound for boosting rounds during tuning and calibration.",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=120,
        help="Early stopping rounds for tuning and full-data calibration.",
    )
    parser.add_argument(
        "--fast-phase-trials",
        type=int,
        default=12,
        help="Number of coarse search trials on a smaller sample. Set to 0 to skip.",
    )
    parser.add_argument(
        "--main-phase-trials",
        type=int,
        default=24,
        help="Number of main trials on a larger sample.",
    )
    parser.add_argument(
        "--fast-phase-train-rows",
        type=int,
        default=120000,
        help="Max train rows for the fast search phase.",
    )
    parser.add_argument(
        "--fast-phase-valid-rows",
        type=int,
        default=60000,
        help="Max valid rows for the fast search phase.",
    )
    parser.add_argument(
        "--max-tune-train-rows",
        type=int,
        default=250000,
        help="Max train rows for the main tuning phase.",
    )
    parser.add_argument(
        "--max-tune-valid-rows",
        type=int,
        default=120000,
        help="Max valid rows for the main tuning phase.",
    )
    parser.add_argument(
        "--trial-parallelism",
        type=int,
        default=1,
        help="How many Optuna trials to run in parallel.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=0,
        help="Threads per LightGBM trial. 0 means auto = cpu_count // trial_parallelism.",
    )
    parser.add_argument(
        "--max-bin",
        type=int,
        default=127,
        help=(
            "Fixed LightGBM max_bin used for all phases. Kept outside Optuna search so "
            "Dataset binning can be reused across trials. Lower values are faster."
        ),
    )
    parser.add_argument(
        "--study-name",
        default="",
        help="Optional Optuna study name. Useful together with --storage for resuming.",
    )
    parser.add_argument(
        "--storage",
        default="",
        help="Optional Optuna storage, e.g. sqlite:///artifacts/lgbm_optuna/study.db",
    )
    return parser.parse_args()


def parse_csv_list(raw_value: str) -> list[str]:
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def infer_metric(task: str, metric: str | None) -> str:
    if metric:
        return metric
    default_metrics = {
        "binary": "auc",
        "multiclass": "multi_logloss",
        "regression": "rmse",
    }
    return default_metrics[task]


def infer_direction(metric: str) -> str:
    maximize_metrics = {"auc", "average_precision", "map", "ndcg", "auc_mu"}
    return "maximize" if metric in maximize_metrics else "minimize"


def resolve_num_threads(num_threads: int, trial_parallelism: int) -> int:
    if num_threads > 0:
        return num_threads
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count // max(1, trial_parallelism))


def lazy_import_dependencies() -> dict[str, Any]:
    try:
        import lightgbm as lgb
        import optuna
        import pandas as pd
        from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error, roc_auc_score
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency. Install at least: pip install lightgbm optuna pandas scikit-learn "
            "(and pyarrow if you read parquet files)."
        ) from exc

    return {
        "lgb": lgb,
        "optuna": optuna,
        "pd": pd,
        "accuracy_score": accuracy_score,
        "log_loss": log_loss,
        "mean_absolute_error": mean_absolute_error,
        "mean_squared_error": mean_squared_error,
        "roc_auc_score": roc_auc_score,
        "train_test_split": train_test_split,
    }


def read_table(path: Path, pd: Any) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".feather":
        return pd.read_feather(path)

    raise ValueError(f"Unsupported file format for {path}. Use CSV, Parquet, or Feather.")


def validate_args(args: argparse.Namespace) -> None:
    use_combined_file = args.data_path is not None
    use_separate_files = args.train_path is not None or args.valid_path is not None or args.oot_path is not None

    if use_combined_file == use_separate_files:
        raise ValueError("Use either --data-path or --train-path/--valid-path/--oot-path, but not both.")
    if use_separate_files and (args.train_path is None or args.valid_path is None):
        raise ValueError("When using separate files, both --train-path and --valid-path are required.")
    if args.fast_phase_trials < 0 or args.main_phase_trials <= 0:
        raise ValueError("Trial counts must satisfy fast_phase_trials >= 0 and main_phase_trials > 0.")
    if args.trial_parallelism <= 0:
        raise ValueError("--trial-parallelism must be positive.")


def load_frames(args: argparse.Namespace, pd: Any) -> tuple[Any, Any, Any | None]:
    if args.data_path is not None:
        df = read_table(args.data_path, pd)
        if args.split_col not in df.columns:
            raise KeyError(f"Split column '{args.split_col}' not found in {args.data_path}.")

        split_values = df[args.split_col].astype("string").str.strip().str.lower()
        train_values = {item.lower() for item in parse_csv_list(args.train_values)}
        valid_values = {item.lower() for item in parse_csv_list(args.valid_values)}
        oot_values = {item.lower() for item in parse_csv_list(args.oot_values)}

        train_df = df.loc[split_values.isin(train_values)].copy()
        valid_df = df.loc[split_values.isin(valid_values)].copy()
        oot_df = df.loc[split_values.isin(oot_values)].copy()
        if oot_df.empty:
            oot_df = None
    else:
        train_df = read_table(args.train_path, pd)
        valid_df = read_table(args.valid_path, pd)
        oot_df = read_table(args.oot_path, pd) if args.oot_path is not None else None

    if train_df.empty:
        raise ValueError("Train split is empty.")
    if valid_df.empty:
        raise ValueError("Valid/test split is empty.")

    return train_df, valid_df, oot_df


def ensure_target_columns(task: str, target_col: str, train_df: Any, valid_df: Any, oot_df: Any | None) -> None:
    for name, frame in (("train", train_df), ("valid", valid_df)):
        if target_col not in frame.columns:
            raise KeyError(f"Target column '{target_col}' not found in {name} split.")

    if oot_df is not None and target_col not in oot_df.columns and task != "regression":
        print("OOT split does not contain target column. OOT evaluation will be skipped.")


def infer_feature_columns(
    train_df: Any,
    valid_df: Any,
    oot_df: Any | None,
    target_col: str,
    split_col: str,
    drop_cols: list[str],
    sample_weight_col: str,
) -> list[str]:
    excluded = {target_col, *drop_cols}
    if split_col in train_df.columns:
        excluded.add(split_col)
    if sample_weight_col:
        excluded.add(sample_weight_col)

    feature_cols = [column for column in train_df.columns if column not in excluded]
    if not feature_cols:
        raise ValueError("No feature columns remain after excluding target/drop/sample_weight columns.")

    missing_in_valid = [column for column in feature_cols if column not in valid_df.columns]
    if missing_in_valid:
        raise KeyError(f"Valid split is missing feature columns: {missing_in_valid}")

    if oot_df is not None:
        missing_in_oot = [column for column in feature_cols if column not in oot_df.columns]
        if missing_in_oot:
            raise KeyError(f"OOT split is missing feature columns: {missing_in_oot}")

    return feature_cols


def build_binary_mapping(labels: list[Any], positive_label: str) -> dict[Any, int]:
    unique_labels = list(dict.fromkeys(label for label in labels if label is not None))
    if len(unique_labels) != 2:
        raise ValueError(f"Binary task expects exactly 2 labels, but got: {unique_labels}")

    if positive_label:
        positive_candidates = [label for label in unique_labels if str(label) == positive_label]
        if len(positive_candidates) != 1:
            raise ValueError(f"Could not match --positive-label={positive_label!r} inside {unique_labels}")
        positive = positive_candidates[0]
        negative = next(label for label in unique_labels if label != positive)
        return {negative: 0, positive: 1}

    normalized = {str(label).strip().lower() for label in unique_labels}
    if normalized in ({"0", "1"}, {"0.0", "1.0"}):
        mapping: dict[Any, int] = {}
        for label in unique_labels:
            mapping[label] = int(float(label))
        return mapping
    if normalized == {"false", "true"}:
        return {label: int(str(label).strip().lower() == "true") for label in unique_labels}

    raise ValueError(
        "Binary target must already be {0,1}/{False,True}, or you must pass --positive-label."
    )


def encode_target_labels(
    task: str,
    target_col: str,
    train_df: Any,
    valid_df: Any,
    oot_df: Any | None,
    positive_label: str,
    pd: Any,
) -> dict[str, Any]:
    if task == "regression":
        return {"label_mapping": {}, "num_class": 0}

    frames_with_target = [train_df, valid_df]
    if oot_df is not None and target_col in oot_df.columns:
        frames_with_target.append(oot_df)

    all_labels = pd.concat([frame[target_col] for frame in frames_with_target], ignore_index=True).dropna().tolist()
    if task == "binary":
        mapping = build_binary_mapping(all_labels, positive_label)
    else:
        unique_labels = sorted(pd.unique(all_labels).tolist(), key=lambda item: str(item))
        mapping = {label: index for index, label in enumerate(unique_labels)}

    for frame in frames_with_target:
        frame[target_col] = frame[target_col].map(mapping)
        if frame[target_col].isna().any():
            raise ValueError("Found target labels that could not be encoded consistently across splits.")
        frame[target_col] = frame[target_col].astype(int)

    return {
        "label_mapping": {str(key): int(value) for key, value in mapping.items()},
        "num_class": len(mapping),
    }


def prepare_categorical_columns(
    train_df: Any,
    valid_df: Any,
    oot_df: Any | None,
    feature_cols: list[str],
    explicit_categorical_cols: list[str],
    pd: Any,
) -> list[str]:
    frames = [train_df, valid_df] + ([oot_df] if oot_df is not None else [])
    auto_categorical_cols: list[str] = []
    for column in feature_cols:
        dtype = train_df[column].dtype
        if (
            pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
        ):
            auto_categorical_cols.append(column)

    categorical_cols = sorted({*explicit_categorical_cols, *auto_categorical_cols}.intersection(feature_cols))
    for column in categorical_cols:
        categories = (
            pd.concat([frame[column].astype("string") for frame in frames if frame is not None], ignore_index=True)
            .dropna()
            .unique()
            .tolist()
        )
        for frame in frames:
            if frame is None:
                continue
            frame[column] = pd.Categorical(frame[column].astype("string"), categories=categories)

    return categorical_cols


def sample_dataframe(
    df: Any,
    task: str,
    target_col: str,
    max_rows: int,
    seed: int,
    train_test_split: Any,
) -> Any:
    if max_rows <= 0 or len(df) <= max_rows:
        return df.copy()

    if task in {"binary", "multiclass"}:
        try:
            sampled_df, _ = train_test_split(
                df,
                train_size=max_rows,
                random_state=seed,
                stratify=df[target_col],
            )
            return sampled_df.reset_index(drop=True)
        except ValueError:
            pass

    return df.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def build_dataset(
    frame: Any,
    feature_cols: list[str],
    target_col: str,
    categorical_cols: list[str],
    sample_weight_col: str,
    lgb: Any,
    reference: Any | None = None,
    dataset_params: dict[str, Any] | None = None,
) -> Any:
    weight = frame[sample_weight_col] if sample_weight_col else None
    dataset = lgb.Dataset(
        frame.loc[:, feature_cols],
        label=frame[target_col],
        weight=weight,
        categorical_feature=categorical_cols or "auto",
        params=dataset_params or None,
        free_raw_data=False,
        reference=reference,
    )
    dataset.construct()
    return dataset


def suggest_search_params(trial: Any) -> dict[str, Any]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
        "max_depth": trial.suggest_categorical("max_depth", [-1, 6, 8, 10, 12]),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 100, 5000, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 0.95),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 0.95),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),
        "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", 1e-3, 10.0, log=True),
        "extra_trees": trial.suggest_categorical("extra_trees", [False, True]),
    }


def build_lgb_params(
    search_params: dict[str, Any],
    task: str,
    metric: str,
    seed: int,
    num_threads: int,
    num_class: int,
    max_bin: int = 127,
) -> dict[str, Any]:
    max_depth = int(search_params["max_depth"])
    raw_num_leaves = int(search_params["num_leaves"])
    num_leaves = raw_num_leaves if max_depth == -1 else min(raw_num_leaves, 2**max_depth)

    params: dict[str, Any] = {
        "objective": task,
        "metric": metric,
        "boosting_type": "gbdt",
        "verbosity": -1,
        "force_col_wise": True,
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "drop_seed": seed,
        "num_threads": num_threads,
        "max_bin": int(max_bin),
        "learning_rate": float(search_params["learning_rate"]),
        "max_depth": max_depth,
        "num_leaves": num_leaves,
        "min_data_in_leaf": int(search_params["min_data_in_leaf"]),
        "feature_fraction": float(search_params["feature_fraction"]),
        "bagging_fraction": float(search_params["bagging_fraction"]),
        "bagging_freq": int(search_params["bagging_freq"]),
        "lambda_l1": float(search_params["lambda_l1"]),
        "lambda_l2": float(search_params["lambda_l2"]),
        "min_gain_to_split": float(search_params["min_gain_to_split"]),
        "min_sum_hessian_in_leaf": float(search_params["min_sum_hessian_in_leaf"]),
        "extra_trees": bool(search_params["extra_trees"]),
    }
    if task == "multiclass":
        params["num_class"] = num_class
    return params


def build_pruning_callback(optuna: Any, trial: Any, metric: str, valid_name: str = "valid") -> Any:
    def callback(env: Any) -> None:
        for data_name, eval_name, value, _higher_is_better in env.evaluation_result_list:
            if data_name == valid_name and eval_name == metric:
                trial.report(float(value), step=env.iteration)
                if trial.should_prune():
                    raise optuna.TrialPruned(f"Pruned at iteration {env.iteration}")
                break

    callback.order = 25
    return callback


def create_objective(
    train_set: Any,
    valid_set: Any,
    task: str,
    metric: str,
    seed: int,
    num_threads: int,
    num_boost_round: int,
    early_stopping_rounds: int,
    num_class: int,
    max_bin: int,
    lgb: Any,
    optuna: Any,
) -> Any:
    def objective(trial: Any) -> float:
        search_params = suggest_search_params(trial)
        params = build_lgb_params(
            search_params=search_params,
            task=task,
            metric=metric,
            seed=seed,
            num_threads=num_threads,
            num_class=num_class,
            max_bin=max_bin,
        )
        booster = lgb.train(
            params=params,
            train_set=train_set,
            num_boost_round=num_boost_round,
            valid_sets=[valid_set],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, first_metric_only=True, verbose=False),
                lgb.log_evaluation(period=0),
                build_pruning_callback(optuna, trial, metric=metric),
            ],
        )
        score = float(booster.best_score["valid"][metric])
        trial.set_user_attr("best_iteration", int(booster.best_iteration))
        trial.set_user_attr("resolved_params", params)
        return score

    return objective


def create_study(optuna: Any, direction: str, seed: int, study_name: str, storage: str) -> Any:
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=50,
        interval_steps=20,
    )
    return optuna.create_study(
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        study_name=study_name or None,
        storage=storage or None,
        load_if_exists=bool(study_name and storage),
    )


def get_best_completed_trial(study: Any, optuna: Any) -> Any:
    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        raise RuntimeError("No completed Optuna trials. Increase trial count or relax pruning.")

    reverse = study.direction.name == "MAXIMIZE"
    completed_trials.sort(key=lambda trial: float(trial.value), reverse=reverse)
    return completed_trials[0]


def get_top_completed_trials(study: Any, optuna: Any, top_k: int) -> list[Any]:
    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    reverse = study.direction.name == "MAXIMIZE"
    completed_trials.sort(key=lambda trial: float(trial.value), reverse=reverse)
    return completed_trials[:top_k]


def run_phase(
    *,
    phase_name: str,
    train_df: Any,
    valid_df: Any,
    feature_cols: list[str],
    target_col: str,
    categorical_cols: list[str],
    sample_weight_col: str,
    task: str,
    metric: str,
    num_class: int,
    max_train_rows: int,
    max_valid_rows: int,
    trials: int,
    seed: int,
    num_threads: int,
    num_boost_round: int,
    early_stopping_rounds: int,
    trial_parallelism: int,
    study_name: str,
    storage: str,
    seed_trials: list[dict[str, Any]] | None,
    dataset_params: dict[str, Any],
    max_bin: int,
    show_progress: bool,
    lgb: Any,
    optuna: Any,
    train_test_split: Any,
) -> tuple[Any, dict[str, Any]]:
    tune_train_df = sample_dataframe(
        df=train_df,
        task=task,
        target_col=target_col,
        max_rows=max_train_rows,
        seed=seed,
        train_test_split=train_test_split,
    )
    tune_valid_df = sample_dataframe(
        df=valid_df,
        task=task,
        target_col=target_col,
        max_rows=max_valid_rows,
        seed=seed + 1,
        train_test_split=train_test_split,
    )

    train_set = build_dataset(
        frame=tune_train_df,
        feature_cols=feature_cols,
        target_col=target_col,
        categorical_cols=categorical_cols,
        sample_weight_col=sample_weight_col,
        lgb=lgb,
        dataset_params=dataset_params,
    )
    valid_set = build_dataset(
        frame=tune_valid_df,
        feature_cols=feature_cols,
        target_col=target_col,
        categorical_cols=categorical_cols,
        sample_weight_col=sample_weight_col,
        lgb=lgb,
        reference=train_set,
        dataset_params=dataset_params,
    )

    direction = infer_direction(metric)
    resolved_study_name = f"{study_name}_{phase_name}" if study_name else ""
    study = create_study(optuna, direction=direction, seed=seed, study_name=resolved_study_name, storage=storage)
    if seed_trials:
        seen: set[str] = set()
        for trial_params in seed_trials:
            dedupe_key = json.dumps(trial_params, sort_keys=True, default=str)
            if dedupe_key in seen:
                continue
            study.enqueue_trial(trial_params)
            seen.add(dedupe_key)
    objective = create_objective(
        train_set=train_set,
        valid_set=valid_set,
        task=task,
        metric=metric,
        seed=seed,
        num_threads=num_threads,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        num_class=num_class,
        max_bin=max_bin,
        lgb=lgb,
        optuna=optuna,
    )
    progress_reporter = _StudyProgressReporter(
        phase_name=phase_name,
        total_trials=trials,
        direction=direction,
        enabled=show_progress,
    )
    study.optimize(
        objective,
        n_trials=trials,
        n_jobs=trial_parallelism,
        callbacks=[progress_reporter],
        gc_after_trial=True,
    )
    progress_reporter.close()

    best_trial = get_best_completed_trial(study, optuna)
    best_params = best_trial.user_attrs.get(
        "resolved_params",
        build_lgb_params(
            search_params=dict(best_trial.params),
            task=task,
            metric=metric,
            seed=seed,
            num_threads=num_threads,
            num_class=num_class,
            max_bin=max_bin,
        ),
    )
    phase_summary = {
        "phase_name": phase_name,
        "sampled_train_rows": int(len(tune_train_df)),
        "sampled_valid_rows": int(len(tune_valid_df)),
        "completed_trials": len(
            [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
        ),
        "best_value": float(best_trial.value),
        "best_iteration": int(best_trial.user_attrs["best_iteration"]),
        "best_search_params": dict(best_trial.params),
        "best_params": best_params,
    }
    return study, phase_summary


def calibrate_best_iteration(
    train_df: Any,
    valid_df: Any,
    feature_cols: list[str],
    target_col: str,
    categorical_cols: list[str],
    sample_weight_col: str,
    params: dict[str, Any],
    num_boost_round: int,
    early_stopping_rounds: int,
    metric: str,
    dataset_params: dict[str, Any],
    lgb: Any,
) -> tuple[int, float]:
    train_set = build_dataset(
        frame=train_df,
        feature_cols=feature_cols,
        target_col=target_col,
        categorical_cols=categorical_cols,
        sample_weight_col=sample_weight_col,
        lgb=lgb,
        dataset_params=dataset_params,
    )
    valid_set = build_dataset(
        frame=valid_df,
        feature_cols=feature_cols,
        target_col=target_col,
        categorical_cols=categorical_cols,
        sample_weight_col=sample_weight_col,
        lgb=lgb,
        reference=train_set,
        dataset_params=dataset_params,
    )
    booster = lgb.train(
        params=params,
        train_set=train_set,
        num_boost_round=num_boost_round,
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, first_metric_only=True, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    return int(booster.best_iteration), float(booster.best_score["valid"][metric])


def fit_final_model(
    train_df: Any,
    valid_df: Any,
    feature_cols: list[str],
    target_col: str,
    categorical_cols: list[str],
    sample_weight_col: str,
    params: dict[str, Any],
    num_boost_round: int,
    dataset_params: dict[str, Any],
    pd: Any,
    lgb: Any,
) -> Any:
    full_train_df = pd.concat([train_df, valid_df], ignore_index=True)
    full_train_set = build_dataset(
        frame=full_train_df,
        feature_cols=feature_cols,
        target_col=target_col,
        categorical_cols=categorical_cols,
        sample_weight_col=sample_weight_col,
        lgb=lgb,
        dataset_params=dataset_params,
    )
    return lgb.train(
        params=params,
        train_set=full_train_set,
        num_boost_round=num_boost_round,
        callbacks=[lgb.log_evaluation(period=0)],
    )


def evaluate_predictions(
    task: str,
    metric: str,
    y_true: Any,
    y_pred: Any,
    accuracy_score: Any,
    log_loss: Any,
    mean_absolute_error: Any,
    mean_squared_error: Any,
    roc_auc_score: Any,
) -> float | None:
    if task == "binary":
        if metric == "auc":
            return float(roc_auc_score(y_true, y_pred))
        if metric == "binary_logloss":
            return float(log_loss(y_true, y_pred, labels=[0, 1]))
        if metric in {"binary_error", "error"}:
            return float(1.0 - accuracy_score(y_true, (y_pred >= 0.5).astype(int)))
        return None

    if task == "multiclass":
        if metric == "multi_logloss":
            return float(log_loss(y_true, y_pred))
        if metric in {"multi_error", "error"}:
            return float(1.0 - accuracy_score(y_true, y_pred.argmax(axis=1)))
        return None

    if metric == "rmse":
        return float(mean_squared_error(y_true, y_pred, squared=False))
    if metric in {"l2", "mse"}:
        return float(mean_squared_error(y_true, y_pred, squared=True))
    if metric in {"l1", "mae"}:
        return float(mean_absolute_error(y_true, y_pred))
    return None


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def save_trials_csv(study: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(output_path, index=False)


def split_dataframe_by_column(
    df: Any,
    split_col: str = "split",
    *,
    train_values: Sequence[str] = (DEFAULT_TRAIN_VALUES,),
    valid_values: Sequence[str] = ("valid", "test", "dev"),
    oot_values: Sequence[str] = (DEFAULT_OOT_VALUES,),
) -> tuple[Any, Any, Any | None]:
    """Split a combined dataframe into train/valid/oot by a split column."""
    if split_col not in df.columns:
        raise KeyError(f"Split column '{split_col}' not found in dataframe.")

    split_series = df[split_col].astype("string").str.strip().str.lower()
    train_value_set = {str(value).strip().lower() for value in train_values}
    valid_value_set = {str(value).strip().lower() for value in valid_values}
    oot_value_set = {str(value).strip().lower() for value in oot_values}

    train_df = df.loc[split_series.isin(train_value_set)].copy()
    valid_df = df.loc[split_series.isin(valid_value_set)].copy()
    oot_df = df.loc[split_series.isin(oot_value_set)].copy()
    if oot_df.empty:
        oot_df = None

    if train_df.empty:
        raise ValueError("Train split is empty after split_dataframe_by_column().")
    if valid_df.empty:
        raise ValueError("Valid split is empty after split_dataframe_by_column().")

    return train_df, valid_df, oot_df


def run_lightgbm_optuna_from_df(
    df: Any,
    *,
    target_col: str,
    split_col: str = "split",
    train_values: Sequence[str] = (DEFAULT_TRAIN_VALUES,),
    valid_values: Sequence[str] = ("valid", "test", "dev"),
    oot_values: Sequence[str] = (DEFAULT_OOT_VALUES,),
    **kwargs: Any,
) -> LightGBMOptunaResult:
    """Notebook-friendly wrapper for a single dataframe containing split labels."""
    train_df, valid_df, oot_df = split_dataframe_by_column(
        df,
        split_col=split_col,
        train_values=train_values,
        valid_values=valid_values,
        oot_values=oot_values,
    )
    return run_lightgbm_optuna(
        train_df=train_df,
        valid_df=valid_df,
        oot_df=oot_df,
        target_col=target_col,
        split_col=split_col,
        **kwargs,
    )


def run_lightgbm_optuna(
    *,
    train_df: Any,
    valid_df: Any,
    target_col: str,
    task: str = "binary",
    oot_df: Any | None = None,
    split_col: str | None = None,
    metric: str | None = None,
    drop_cols: Sequence[str] = (),
    categorical_cols: Sequence[str] = (),
    sample_weight_col: str = "",
    positive_label: str = "",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    seed: int = 42,
    num_boost_round: int = 3000,
    early_stopping_rounds: int = 120,
    fast_phase_trials: int = 12,
    main_phase_trials: int = 24,
    fast_phase_train_rows: int = 120000,
    fast_phase_valid_rows: int = 60000,
    max_tune_train_rows: int = 250000,
    max_tune_valid_rows: int = 120000,
    trial_parallelism: int = 1,
    num_threads: int = 0,
    max_bin: int = 127,
    study_name: str = "",
    storage: str = "",
    save_artifacts: bool = True,
    verbose: bool = True,
    show_progress: bool = True,
) -> LightGBMOptunaResult:
    """Run LightGBM + Optuna tuning directly from dataframes.

    This is the notebook-friendly API. Pass prepared train/valid/oot dataframes and
    receive the fitted model, params, summaries, studies, and artifact paths.
    """
    if fast_phase_trials < 0 or main_phase_trials <= 0:
        raise ValueError("Trial counts must satisfy fast_phase_trials >= 0 and main_phase_trials > 0.")
    if trial_parallelism <= 0:
        raise ValueError("trial_parallelism must be positive.")

    deps = lazy_import_dependencies()
    lgb = deps["lgb"]
    optuna = deps["optuna"]
    pd = deps["pd"]
    train_test_split = deps["train_test_split"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    metric = infer_metric(task, metric)
    num_threads = resolve_num_threads(num_threads, trial_parallelism)
    dataset_params = {"max_bin": int(max_bin), "feature_pre_filter": False}
    output_dir = Path(output_dir).expanduser().resolve()
    if save_artifacts:
        output_dir.mkdir(parents=True, exist_ok=True)

    working_train_df = train_df.copy()
    working_valid_df = valid_df.copy()
    working_oot_df = oot_df.copy() if oot_df is not None else None

    ensure_target_columns(task, target_col, working_train_df, working_valid_df, working_oot_df)

    drop_cols_list = list(drop_cols)
    categorical_cols_hint = list(categorical_cols)
    sample_weight_col = sample_weight_col.strip()
    positive_label = positive_label.strip()

    label_info = encode_target_labels(
        task=task,
        target_col=target_col,
        train_df=working_train_df,
        valid_df=working_valid_df,
        oot_df=working_oot_df,
        positive_label=positive_label,
        pd=pd,
    )
    feature_cols = infer_feature_columns(
        train_df=working_train_df,
        valid_df=working_valid_df,
        oot_df=working_oot_df,
        target_col=target_col,
        split_col=split_col or "",
        drop_cols=drop_cols_list,
        sample_weight_col=sample_weight_col,
    )
    resolved_categorical_cols = prepare_categorical_columns(
        train_df=working_train_df,
        valid_df=working_valid_df,
        oot_df=working_oot_df,
        feature_cols=feature_cols,
        explicit_categorical_cols=categorical_cols_hint,
        pd=pd,
    )

    phase_summaries: list[dict[str, Any]] = []
    phase_studies: dict[str, Any | None] = {"phase1_fast": None, "phase2_main": None}
    phase2_seed_trials: list[dict[str, Any]] | None = None

    phase1_trials_path = output_dir / "phase1_trials.csv"
    phase2_trials_path = output_dir / "phase2_trials.csv"
    best_params_path = output_dir / "best_params.json"
    summary_path = output_dir / "summary.json"
    model_path = output_dir / "lightgbm_model.txt"

    if fast_phase_trials > 0:
        if verbose:
            print(
                f"[phase1_fast] tuning on train={min(len(working_train_df), fast_phase_train_rows)} "
                f"valid={min(len(working_valid_df), fast_phase_valid_rows)} rows, "
                f"trials={fast_phase_trials}"
            )
        phase1_study, phase1_summary = run_phase(
            phase_name="phase1_fast",
            train_df=working_train_df,
            valid_df=working_valid_df,
            feature_cols=feature_cols,
            target_col=target_col,
            categorical_cols=resolved_categorical_cols,
            sample_weight_col=sample_weight_col,
            task=task,
            metric=metric,
            num_class=label_info["num_class"],
            max_train_rows=fast_phase_train_rows,
            max_valid_rows=fast_phase_valid_rows,
            trials=fast_phase_trials,
            seed=seed,
            num_threads=num_threads,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            trial_parallelism=trial_parallelism,
            study_name=study_name,
            storage=storage,
            seed_trials=None,
            dataset_params=dataset_params,
            max_bin=max_bin,
            show_progress=show_progress,
            lgb=lgb,
            optuna=optuna,
            train_test_split=train_test_split,
        )
        phase_studies["phase1_fast"] = phase1_study
        phase_summaries.append(phase1_summary)
        if save_artifacts:
            save_trials_csv(phase1_study, phase1_trials_path)
        phase2_seed_trials = [dict(trial.params) for trial in get_top_completed_trials(phase1_study, optuna, top_k=5)]

    if verbose:
        print(
            f"[phase2_main] tuning on train={min(len(working_train_df), max_tune_train_rows)} "
            f"valid={min(len(working_valid_df), max_tune_valid_rows)} rows, "
            f"trials={main_phase_trials}"
        )
    phase2_study, phase2_summary = run_phase(
        phase_name="phase2_main",
        train_df=working_train_df,
        valid_df=working_valid_df,
        feature_cols=feature_cols,
        target_col=target_col,
        categorical_cols=resolved_categorical_cols,
        sample_weight_col=sample_weight_col,
        task=task,
        metric=metric,
        num_class=label_info["num_class"],
        max_train_rows=max_tune_train_rows,
        max_valid_rows=max_tune_valid_rows,
        trials=main_phase_trials,
        seed=seed + 100,
        num_threads=num_threads,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        trial_parallelism=trial_parallelism,
        study_name=study_name,
        storage=storage,
        seed_trials=phase2_seed_trials,
        dataset_params=dataset_params,
        max_bin=max_bin,
        show_progress=show_progress,
        lgb=lgb,
        optuna=optuna,
        train_test_split=train_test_split,
    )
    phase_studies["phase2_main"] = phase2_study
    phase_summaries.append(phase2_summary)
    if save_artifacts:
        save_trials_csv(phase2_study, phase2_trials_path)

    best_trial = get_best_completed_trial(phase2_study, optuna)
    best_params = build_lgb_params(
        search_params=dict(best_trial.params),
        task=task,
        metric=metric,
        seed=seed,
        num_threads=num_threads,
        num_class=label_info["num_class"],
        max_bin=max_bin,
    )

    calibrated_best_iteration, calibration_score = calibrate_best_iteration(
        train_df=working_train_df,
        valid_df=working_valid_df,
        feature_cols=feature_cols,
        target_col=target_col,
        categorical_cols=resolved_categorical_cols,
        sample_weight_col=sample_weight_col,
        params=best_params,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        metric=metric,
        dataset_params=dataset_params,
        lgb=lgb,
    )
    final_model = fit_final_model(
        train_df=working_train_df,
        valid_df=working_valid_df,
        feature_cols=feature_cols,
        target_col=target_col,
        categorical_cols=resolved_categorical_cols,
        sample_weight_col=sample_weight_col,
        params=best_params,
        num_boost_round=calibrated_best_iteration,
        dataset_params=dataset_params,
        pd=pd,
        lgb=lgb,
    )
    if save_artifacts:
        final_model.save_model(str(model_path))

    oot_score = None
    if working_oot_df is not None and target_col in working_oot_df.columns:
        oot_pred = final_model.predict(
            working_oot_df.loc[:, feature_cols],
            num_iteration=final_model.current_iteration(),
        )
        oot_score = evaluate_predictions(
            task=task,
            metric=metric,
            y_true=working_oot_df[target_col],
            y_pred=oot_pred,
            accuracy_score=deps["accuracy_score"],
            log_loss=deps["log_loss"],
            mean_absolute_error=deps["mean_absolute_error"],
            mean_squared_error=deps["mean_squared_error"],
            roc_auc_score=deps["roc_auc_score"],
        )

    artifact_paths = {
        "model_path": str(model_path) if save_artifacts else None,
        "best_params_path": str(best_params_path) if save_artifacts else None,
        "summary_path": str(summary_path) if save_artifacts else None,
        "phase1_trials_csv": str(phase1_trials_path) if save_artifacts and fast_phase_trials > 0 else None,
        "phase2_trials_csv": str(phase2_trials_path) if save_artifacts else None,
    }
    summary = {
        "task": task,
        "metric": metric,
        "seed": seed,
        "num_threads": num_threads,
        "trial_parallelism": trial_parallelism,
        "rows": {
            "train": int(len(working_train_df)),
            "valid": int(len(working_valid_df)),
            "oot": int(len(working_oot_df)) if working_oot_df is not None else 0,
        },
        "feature_count": len(feature_cols),
        "categorical_feature_count": len(resolved_categorical_cols),
        "categorical_features": resolved_categorical_cols,
        "label_mapping": label_info["label_mapping"],
        "phases": phase_summaries,
        "best_trial_number": int(best_trial.number),
        "best_tuning_value": float(best_trial.value),
        "calibration_score": calibration_score,
        "calibrated_best_iteration": int(calibrated_best_iteration),
        "oot_score": oot_score,
        "artifacts": artifact_paths,
    }

    if save_artifacts:
        save_json(
            best_params_path,
            {
                "best_trial_number": int(best_trial.number),
                "metric": metric,
                "best_value": float(best_trial.value),
                "calibrated_best_iteration": int(calibrated_best_iteration),
                "params": best_params,
            },
        )
        save_json(summary_path, summary)

    if verbose:
        if save_artifacts:
            print(f"Done. Model saved to: {model_path}")
            print(f"Summary saved to: {summary_path}")
        print(f"Best metric on sampled tuning set: {best_trial.value:.6f}")
        print(f"Best iteration calibrated on full train/valid: {calibrated_best_iteration}")
        if oot_score is not None:
            print(f"OOT {metric}: {oot_score:.6f}")

    return LightGBMOptunaResult(
        model=final_model,
        best_params=best_params,
        summary=summary,
        feature_cols=feature_cols,
        categorical_cols=resolved_categorical_cols,
        label_mapping=label_info["label_mapping"],
        phase_summaries=phase_summaries,
        studies=phase_studies,
        artifact_paths=artifact_paths,
        best_trial_number=int(best_trial.number),
        best_tuning_value=float(best_trial.value),
        calibrated_best_iteration=int(calibrated_best_iteration),
        calibration_score=float(calibration_score),
        oot_score=None if oot_score is None else float(oot_score),
    )


def run_lightgbm_optuna_from_split_df(
    *,
    df: Any,
    target_col: str,
    split_col: str = "split",
    train_values: Sequence[str] = (DEFAULT_TRAIN_VALUES,),
    valid_values: Sequence[str] = ("valid", "test", "dev"),
    oot_values: Sequence[str] = (DEFAULT_OOT_VALUES,),
    **kwargs: Any,
) -> LightGBMOptunaResult:
    """Alias of run_lightgbm_optuna_from_df() with a more explicit name."""
    return run_lightgbm_optuna_from_df(
        df=df,
        target_col=target_col,
        split_col=split_col,
        train_values=train_values,
        valid_values=valid_values,
        oot_values=oot_values,
        **kwargs,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)

    pd = lazy_import_dependencies()["pd"]
    train_df, valid_df, oot_df = load_frames(args, pd)
    run_lightgbm_optuna(
        train_df=train_df,
        valid_df=valid_df,
        oot_df=oot_df,
        target_col=args.target_col,
        task=args.task,
        split_col=args.split_col if args.data_path is not None else None,
        metric=args.metric,
        drop_cols=parse_csv_list(args.drop_cols),
        categorical_cols=parse_csv_list(args.categorical_cols),
        sample_weight_col=args.sample_weight_col,
        positive_label=args.positive_label,
        output_dir=args.output_dir,
        seed=args.seed,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        fast_phase_trials=args.fast_phase_trials,
        main_phase_trials=args.main_phase_trials,
        fast_phase_train_rows=args.fast_phase_train_rows,
        fast_phase_valid_rows=args.fast_phase_valid_rows,
        max_tune_train_rows=args.max_tune_train_rows,
        max_tune_valid_rows=args.max_tune_valid_rows,
        trial_parallelism=args.trial_parallelism,
        num_threads=args.num_threads,
        max_bin=args.max_bin,
        study_name=args.study_name,
        storage=args.storage,
        save_artifacts=True,
        verbose=True,
        show_progress=True,
    )


__all__ = [
    "LightGBMOptunaResult",
    "build_lgb_params",
    "infer_direction",
    "infer_metric",
    "resolve_num_threads",
    "run_lightgbm_optuna",
    "run_lightgbm_optuna_from_df",
    "run_lightgbm_optuna_from_split_df",
    "split_dataframe_by_column",
]


if __name__ == "__main__":
    main()
