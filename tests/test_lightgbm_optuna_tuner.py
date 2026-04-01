import io
import unittest
from contextlib import redirect_stdout

import pandas as pd

from lightgbm_optuna_tuner import (
    TQDM_AVAILABLE,
    build_lgb_params,
    infer_direction,
    infer_metric,
    infer_feature_columns,
    resolve_num_threads,
    split_dataframe_by_column,
    _StudyProgressReporter,
)


class LightGBMOptunaTunerTests(unittest.TestCase):
    def test_build_lgb_params_caps_num_leaves_by_max_depth(self) -> None:
        params = build_lgb_params(
            search_params={
                "learning_rate": 0.05,
                "max_depth": 6,
                "num_leaves": 255,
                "min_data_in_leaf": 200,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.9,
                "bagging_freq": 1,
                "lambda_l1": 0.0,
                "lambda_l2": 0.0,
                "min_gain_to_split": 0.0,
                "min_sum_hessian_in_leaf": 0.001,
                "max_bin": 127,
                "extra_trees": True,
            },
            task="binary",
            metric="auc",
            seed=42,
            num_threads=2,
            num_class=0,
        )
        self.assertEqual(params["max_depth"], 6)
        self.assertEqual(params["num_leaves"], 64)
        self.assertTrue(params["extra_trees"])

    def test_metric_helpers_return_expected_defaults(self) -> None:
        self.assertEqual(infer_metric("binary", None), "auc")
        self.assertEqual(infer_metric("multiclass", None), "multi_logloss")
        self.assertEqual(infer_metric("regression", None), "rmse")
        self.assertEqual(infer_direction("auc"), "maximize")
        self.assertEqual(infer_direction("rmse"), "minimize")

    def test_resolve_num_threads_uses_at_least_one_thread(self) -> None:
        self.assertEqual(resolve_num_threads(3, 2), 3)
        self.assertGreaterEqual(resolve_num_threads(0, 999), 1)

    def test_split_dataframe_by_column_returns_expected_frames(self) -> None:
        df = pd.DataFrame(
            [
                {"split": "train", "target": 0},
                {"split": "train", "target": 1},
                {"split": "test", "target": 0},
                {"split": "oot", "target": 1},
            ]
        )
        train_df, valid_df, oot_df = split_dataframe_by_column(
            df,
            split_col="split",
            train_values=("train",),
            valid_values=("test",),
            oot_values=("oot",),
        )
        self.assertEqual(len(train_df), 2)
        self.assertEqual(len(valid_df), 1)
        self.assertIsNotNone(oot_df)
        self.assertEqual(len(oot_df), 1)

    def test_split_dataframe_by_column_without_oot_returns_none(self) -> None:
        df = pd.DataFrame(
            [
                {"split": "train", "target": 0},
                {"split": "valid", "target": 1},
            ]
        )
        train_df, valid_df, oot_df = split_dataframe_by_column(df)
        self.assertEqual(len(train_df), 1)
        self.assertEqual(len(valid_df), 1)
        self.assertIsNone(oot_df)

    def test_infer_feature_columns_excludes_custom_split_col(self) -> None:
        train_df = pd.DataFrame(
            {
                "dataset_flag": ["train", "train"],
                "target": [0, 1],
                "f1": [1.0, 2.0],
                "f2": [3.0, 4.0],
            }
        )
        valid_df = pd.DataFrame(
            {
                "dataset_flag": ["valid"],
                "target": [1],
                "f1": [5.0],
                "f2": [6.0],
            }
        )
        feature_cols = infer_feature_columns(
            train_df=train_df,
            valid_df=valid_df,
            oot_df=None,
            target_col="target",
            split_col="dataset_flag",
            drop_cols=[],
            sample_weight_col="",
        )
        self.assertEqual(feature_cols, ["f1", "f2"])

    def test_study_progress_reporter_prints_text_without_tqdm(self) -> None:
        import lightgbm_optuna_tuner

        previous_value = TQDM_AVAILABLE
        lightgbm_optuna_tuner.TQDM_AVAILABLE = False
        reporter = _StudyProgressReporter(
            phase_name="phase_demo",
            total_trials=3,
            direction="maximize",
            enabled=True,
        )

        class Trial:
            def __init__(self, value: float) -> None:
                self.value = value

        class Study:
            def __init__(self, values: list[float]) -> None:
                self.trials = [Trial(value) for value in values]

        buffer = io.StringIO()
        try:
            with redirect_stdout(buffer):
                reporter(Study([0.61]), None)
                reporter(Study([0.61, 0.63]), None)
                reporter(Study([0.61, 0.63, 0.62]), None)
                reporter.close()
        finally:
            lightgbm_optuna_tuner.TQDM_AVAILABLE = previous_value

        progress_output = buffer.getvalue()
        self.assertIn("phase_demo", progress_output)
        self.assertIn("1/3", progress_output)
        self.assertIn("3/3", progress_output)
        self.assertIn("best=0.630000", progress_output)


if __name__ == "__main__":
    unittest.main()
