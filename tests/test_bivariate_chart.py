import tempfile
import unittest
from pathlib import Path

import pandas as pd

from bivariate_chart import (
    build_bivariate_summaries,
    build_bivariate_summary,
    build_and_plot_bivariate_charts,
)


class BivariateChartTests(unittest.TestCase):
    def test_missing_and_zero_are_kept_as_standalone_bins(self) -> None:
        df = pd.DataFrame(
            {
                "feature": [None, "bad", 0, 0, 1, 2, 3],
                "label": [1, 0, 0, 1, 1, 0, 1],
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
        self.assertEqual(result.summary.iloc[1]["bin"], "0")
        self.assertEqual(int(result.summary.iloc[1]["count"]), 2)
        self.assertFalse(bool(result.summary.iloc[1]["is_missing"]))
        self.assertEqual(result.actual_bin_count, 3)

    def test_interval_labels_use_explicit_open_closed_rules(self) -> None:
        df = pd.DataFrame(
            {
                "feature": [0, 1, 2, 3, 4, 5, 6],
                "label": [0, 0, 1, 0, 1, 1, 0],
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
        self.assertEqual(bins, ["0", "[1, 2.67]", "(2.67, 4.33]", "(4.33, 6]"])
        self.assertTrue(bool(result.summary.iloc[0]["left_closed"]))
        self.assertTrue(bool(result.summary.iloc[0]["right_closed"]))
        self.assertTrue(bool(result.summary.iloc[1]["left_closed"]))
        self.assertTrue(bool(result.summary.iloc[1]["right_closed"]))
        self.assertFalse(bool(result.summary.iloc[2]["left_closed"]))
        self.assertTrue(bool(result.summary.iloc[2]["right_closed"]))
        self.assertIn("[a, b]", result.interval_rule)
        self.assertIn("(b, c]", result.interval_rule)
        self.assertIn("0 is shown as a standalone bin", result.interval_rule)

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

    def test_only_zero_values_keep_zero_as_a_single_bin(self) -> None:
        df = pd.DataFrame(
            {
                "feature": [0, 0, 0, None],
                "label": [1, 0, 1, 0],
            }
        )

        result = build_bivariate_summary(
            df,
            feature_col="feature",
            target_col="label",
            n_bins=4,
        )

        self.assertEqual(result.actual_bin_count, 0)
        self.assertEqual(result.summary["bin"].tolist(), ["Missing", "0"])
        self.assertEqual(result.summary["count"].tolist(), [1, 3])

    def test_build_bivariate_summaries_supports_multiple_features(self) -> None:
        df = pd.DataFrame(
            {
                "feature_a": [None, 0, 1, 2, 3, 4],
                "feature_b": [0, 0, 10, 20, 30, 40],
                "label": [1, 0, 1, 0, 1, 0],
            }
        )

        results = build_bivariate_summaries(
            df,
            feature_cols=["feature_a", "feature_b"],
            target_col="label",
            n_bins=3,
        )

        self.assertEqual(set(results.keys()), {"feature_a", "feature_b"})
        self.assertEqual(results["feature_a"].feature_col, "feature_a")
        self.assertEqual(results["feature_b"].feature_col, "feature_b")
        self.assertIn("0", results["feature_a"].summary["bin"].tolist())
        self.assertIn("0", results["feature_b"].summary["bin"].tolist())

    def test_build_and_plot_bivariate_charts_auto_generates_titles_and_save_paths(self) -> None:
        df = pd.DataFrame(
            {
                "feature_a": [None, 0, 1, 2, 3, 4],
                "feature_b": [0, 0, 10, 20, 30, 40],
                "label": [1, 0, 1, 0, 1, 0],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            batch_results = build_and_plot_bivariate_charts(
                df,
                feature_cols=["feature_a", "feature_b"],
                target_col="label",
                n_bins=3,
                save_dir=temp_dir,
            )

            self.assertEqual(set(batch_results.keys()), {"feature_a", "feature_b"})
            for feature_col, payload in batch_results.items():
                result = payload.result
                figure = payload.figure
                axes = payload.axes
                expected_path = Path(temp_dir) / f"{feature_col}_vs_label_bivariate.png"

                self.assertEqual(result.feature_col, feature_col)
                self.assertTrue(expected_path.exists())
                self.assertEqual(payload.title, f"{feature_col} vs label")
                self.assertEqual(payload.save_path, str(expected_path))
                self.assertEqual(axes[0].get_title(), f"{feature_col} vs label")
                figure.clf()


if __name__ == "__main__":
    unittest.main()
