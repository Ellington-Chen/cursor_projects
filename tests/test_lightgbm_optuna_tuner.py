import unittest

from lightgbm_optuna_tuner import build_lgb_params, infer_direction, infer_metric, resolve_num_threads


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


if __name__ == "__main__":
    unittest.main()
