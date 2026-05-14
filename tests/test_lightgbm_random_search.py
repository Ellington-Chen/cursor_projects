import io
import unittest
from contextlib import redirect_stdout

import pandas as pd

import lightgbm_random_search
from lightgbm_random_search import (
    build_backward_selection_random_search_space,
    infer_random_search_scoring,
    run_lightgbm_random_search_from_df,
)


class LightGBMRandomSearchTests(unittest.TestCase):
    def _build_train_df(self) -> pd.DataFrame:
        row_count = 120
        return pd.DataFrame(
            {
                "target": [index % 2 for index in range(row_count)],
                "f1": [float(index) for index in range(row_count)],
                "f2": [float((index * 3) % 11) for index in range(row_count)],
                "unused": [float(index % 5) for index in range(row_count)],
            }
        )

    def test_metric_helpers_return_expected_defaults(self) -> None:
        self.assertEqual(infer_random_search_scoring("classification"), "roc_auc")
        self.assertEqual(infer_random_search_scoring("regression"), "neg_mean_squared_error")
        self.assertEqual(infer_random_search_scoring("classification", "average_precision"), "average_precision")

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
        train_df = self._build_train_df()

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

    def test_run_lightgbm_random_search_prints_text_progress_without_tqdm(self) -> None:
        train_df = self._build_train_df()
        previous_value = lightgbm_random_search.TQDM_AVAILABLE
        lightgbm_random_search.TQDM_AVAILABLE = False
        buffer = io.StringIO()

        try:
            with redirect_stdout(buffer):
                result = run_lightgbm_random_search_from_df(
                    train_df,
                    target_col="target",
                    feature_cols=["f1", "f2"],
                    n_iter=1,
                    cv=2,
                    search_n_jobs=1,
                    model_n_jobs=1,
                    random_state=7,
                    show_progress=True,
                )
        finally:
            lightgbm_random_search.TQDM_AVAILABLE = previous_value

        progress_output = buffer.getvalue()
        self.assertIsNotNone(result.best_estimator)
        self.assertIn("random_search: starting", progress_output)
        self.assertIn("best_score=", progress_output)
        self.assertIn("refit=done", progress_output)

    def test_run_lightgbm_random_search_can_disable_progress_output(self) -> None:
        train_df = self._build_train_df()
        buffer = io.StringIO()

        with redirect_stdout(buffer):
            result = run_lightgbm_random_search_from_df(
                train_df,
                target_col="target",
                feature_cols=["f1", "f2"],
                n_iter=1,
                cv=2,
                search_n_jobs=1,
                model_n_jobs=1,
                random_state=7,
                show_progress=False,
            )

        self.assertIsNotNone(result.best_estimator)
        self.assertEqual(buffer.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
