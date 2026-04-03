import unittest

import pandas as pd

from bivariate_chart import build_bivariate_summary


class BivariateChartTests(unittest.TestCase):
    def test_missing_is_kept_as_a_standalone_bin(self) -> None:
        df = pd.DataFrame(
            {
                "feature": [None, "bad", 0, 1, 2, 3],
                "label": [1, 0, 0, 1, 0, 1],
            }
        )

        result = build_bivariate_summary(
            df,
            feature_col="feature",
            target_col="label",
            n_bins=3,
        )

        self.assertEqual(result.summary.iloc[0]["bin"], "Missing")
        self.assertEqual(int(result.summary.iloc[0]["count"]), 2)
        self.assertTrue(bool(result.summary.iloc[0]["is_missing"]))
        self.assertEqual(result.actual_bin_count, 3)

    def test_interval_labels_use_explicit_open_closed_rules(self) -> None:
        df = pd.DataFrame(
            {
                "feature": [0, 1, 2, 3, 4, 5],
                "label": [0, 0, 1, 0, 1, 1],
            }
        )

        result = build_bivariate_summary(
            df,
            feature_col="feature",
            target_col="label",
            n_bins=3,
            precision=2,
        )

        bins = result.summary["bin"].tolist()
        self.assertEqual(bins, ["[0, 1.67]", "(1.67, 3.33]", "(3.33, 5]"])
        self.assertTrue(bool(result.summary.iloc[0]["left_closed"]))
        self.assertTrue(bool(result.summary.iloc[0]["right_closed"]))
        self.assertFalse(bool(result.summary.iloc[1]["left_closed"]))
        self.assertTrue(bool(result.summary.iloc[1]["right_closed"]))
        self.assertIn("[a, b]", result.interval_rule)
        self.assertIn("(b, c]", result.interval_rule)

    def test_equal_value_feature_collapses_to_single_non_missing_bin(self) -> None:
        df = pd.DataFrame(
            {
                "feature": [2, 2, 2, None],
                "label": [0, 1, 1, 0],
            }
        )

        result = build_bivariate_summary(
            df,
            feature_col="feature",
            target_col="label",
            n_bins=4,
        )

        self.assertEqual(result.actual_bin_count, 1)
        self.assertEqual(result.summary["bin"].tolist(), ["Missing", "[2, 2]"])
        self.assertEqual(result.summary["count"].tolist(), [1, 3])


if __name__ == "__main__":
    unittest.main()
