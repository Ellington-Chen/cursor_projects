import io
import unittest
from contextlib import redirect_stdout

import pandas as pd

from lightgbm_optuna_tuner import (
    TQDM_AVAILABLE,
    build_backward_selection_random_search_space,
    build_lgb_params,
    infer_direction,
    infer_metric,
    infer_feature_columns,
    infer_random_search_scoring,
    resolve_num_threads,
    run_lightgbm_random_search_from_df,
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
        self.assertEqual(infer_random_search_scoring("classification"), "roc_auc")
        self.assertEqual(infer_random_search_scoring("regression"), "neg_mean_squared_error")
        self.assertEqual(infer_random_search_scoring("classification", "average_precision"), "average_precision")

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

    def test_backward_selection_random_search_space_matches_expected_ranges(self) -> None:
        space = build_backward_selection_random_search_space()

        self.assertEqual(
            set(space.keys()),
            {
                "num_leaves",
                "max_bin",
                "max_depth",
                "n_estimators",
                "scale_pos_weight",
                "learning_rate",
                "min_child_samples",
                "subsample",
                "colsample_bytree",
                "reg_alpha",
            },
        )
        self.assertEqual(space["max_bin"], list(range(200, 2001, 100)))
        self.assertEqual(space["n_estimators"], list(range(20, 301, 20)))
        self.assertEqual(space["scale_pos_weight"], [0.8, 1])
        self.assertEqual(space["min_child_samples"], list(range(2000, 14001, 500)))

        num_leaves_sample = int(space["num_leaves"].rvs(random_state=7))
        max_depth_sample = int(space["max_depth"].rvs(random_state=7))
        learning_rate_sample = float(space["learning_rate"].rvs(random_state=7))
        subsample_sample = float(space["subsample"].rvs(random_state=7))
        colsample_sample = float(space["colsample_bytree"].rvs(random_state=7))
        reg_alpha_sample = float(space["reg_alpha"].rvs(random_state=7))

        self.assertGreaterEqual(num_leaves_sample, 4)
        self.assertLess(num_leaves_sample, 10)
        self.assertGreaterEqual(max_depth_sample, 2)
        self.assertLess(max_depth_sample, 4)
        self.assertGreaterEqual(learning_rate_sample, 0.004)
        self.assertLess(learning_rate_sample, 0.054)
        self.assertGreaterEqual(subsample_sample, 0.7)
        self.assertLess(subsample_sample, 1.0)
        self.assertGreaterEqual(colsample_sample, 0.7)
        self.assertLess(colsample_sample, 1.0)
        self.assertGreaterEqual(reg_alpha_sample, 0.0001)
        self.assertLess(reg_alpha_sample, 10.0001)

    def test_run_lightgbm_random_search_from_df_runs_single_search_round(self) -> None:
        row_count = 120
        train_df = pd.DataFrame(
            {
                "target": [index % 2 for index in range(row_count)],
                "f1": [float(index) for index in range(row_count)],
                "f2": [float((index * 3) % 11) for index in range(row_count)],
                "unused": [float(index % 5) for index in range(row_count)],
            }
        )

        result = run_lightgbm_random_search_from_df(
            train_df,
            target_col="target",
            feature_cols=["f1", "f2"],
            n_iter=1,
            cv=2,
            search_n_jobs=1,
            model_n_jobs=1,
            random_state=7,
        )

        self.assertEqual(result.feature_cols, ["f1", "f2"])
        self.assertEqual(result.scoring, "roc_auc")
        self.assertEqual(set(result.best_params.keys()), set(build_backward_selection_random_search_space().keys()))
        self.assertIsNotNone(result.best_estimator)
        self.assertEqual(result.best_model_params["objective"], "binary")
        self.assertEqual(result.best_model_params["metric"], "auc")
        self.assertEqual(result.best_model_params["n_jobs"], 1)
        self.assertIsInstance(result.best_score, float)

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
