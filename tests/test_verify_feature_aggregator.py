import unittest

import pandas as pd

from verify_feature_aggregator import (
    add_verify_all_source_group_features,
    build_verify_feature_name,
    list_verify_all_feature_columns,
    list_verify_source_feature_columns,
)


class VerifyFeatureAggregatorTests(unittest.TestCase):
    def test_default_feature_name_counts_match_expected_shapes(self) -> None:
        source_columns = list_verify_source_feature_columns()
        all_columns = list_verify_all_feature_columns()

        self.assertEqual(len(source_columns), 1350)
        self.assertEqual(len(all_columns), 270)

    def test_add_verify_all_source_group_features_sums_five_source_groups(self) -> None:
        row_one_values = {
            "bank": 1,
            "cf": 2,
            "top": 3,
            "others": 4,
            "nonloan": 5,
        }
        row_two_values = {
            "bank": 10,
            "cf": 20,
            "top": 30,
            "others": 40,
            "nonloan": 50,
        }
        df = pd.DataFrame(
            [
                {
                    build_verify_feature_name("debit", "bank", "succ", "ordercnt", 3): row_one_values["bank"],
                    build_verify_feature_name("debit", "cf", "succ", "ordercnt", 3): row_one_values["cf"],
                    build_verify_feature_name("debit", "top", "succ", "ordercnt", 3): row_one_values["top"],
                    build_verify_feature_name("debit", "others", "succ", "ordercnt", 3): row_one_values["others"],
                    build_verify_feature_name("debit", "nonloan", "succ", "ordercnt", 3): row_one_values["nonloan"],
                },
                {
                    build_verify_feature_name("debit", "bank", "succ", "ordercnt", 3): row_two_values["bank"],
                    build_verify_feature_name("debit", "cf", "succ", "ordercnt", 3): row_two_values["cf"],
                    build_verify_feature_name("debit", "top", "succ", "ordercnt", 3): row_two_values["top"],
                    build_verify_feature_name("debit", "others", "succ", "ordercnt", 3): row_two_values["others"],
                    build_verify_feature_name("debit", "nonloan", "succ", "ordercnt", 3): row_two_values["nonloan"],
                },
            ]
        )

        result = add_verify_all_source_group_features(
            df,
            card_types=("debit",),
            result_types=("succ",),
            metrics=("ordercnt",),
            day_windows=(3,),
        )

        target_col = build_verify_feature_name("debit", "all", "succ", "ordercnt", 3)
        self.assertIn(target_col, result.columns)
        self.assertEqual(result[target_col].tolist(), [15, 150])

    def test_non_strict_mode_treats_missing_source_columns_as_zero(self) -> None:
        bank_col = build_verify_feature_name("credit", "bank", "fail", "cardcnt", 7)
        cf_col = build_verify_feature_name("credit", "cf", "fail", "cardcnt", 7)
        target_col = build_verify_feature_name("credit", "all", "fail", "cardcnt", 7)

        df = pd.DataFrame([{bank_col: 6, cf_col: 4}])

        result = add_verify_all_source_group_features(
            df,
            card_types=("credit",),
            result_types=("fail",),
            metrics=("cardcnt",),
            day_windows=(7,),
            strict=False,
        )

        self.assertEqual(result[target_col].tolist(), [10])


if __name__ == "__main__":
    unittest.main()
