import unittest

import pandas as pd

from verify_feature_utils import (
    VERIFY_IMAGE_DAY_WINDOWS,
    VERIFY_IMAGE_FIRST_BIND_MONTHS_COLUMN,
    VERIFY_IMAGE_LAST_BIND_MONTHS_COLUMN,
    VERIFY_IMAGE_METRICS,
    VERIFY_IMAGE_SOURCE_GROUPS,
    build_verify_first_bind_months_feature_name,
    build_verify_feature_name,
    build_verify_image_variables,
    build_verify_last_bind_months_feature_name,
    transform_verify_features,
)


def _build_source_row(**overrides: int | float) -> dict[str, int | float]:
    row = {
        build_verify_feature_name("all", source_group, "all", metric, day_window): 0
        for source_group in VERIFY_IMAGE_SOURCE_GROUPS
        for metric in VERIFY_IMAGE_METRICS
        for day_window in VERIFY_IMAGE_DAY_WINDOWS
    }
    row.update(overrides)
    return row


class VerifyFeatureUtilsTests(unittest.TestCase):
    def test_build_verify_image_variables_returns_only_plot_columns(self) -> None:
        df = pd.DataFrame(
            [
                _build_source_row(
                    verify_all_bank_all_membercnt_day31=1,
                    verify_all_cf_all_membercnt_day60=2,
                    verify_all_top_all_membercnt_day180=3,
                    verify_all_others_all_ordercnt_day31=4,
                    verify_all_nonloan_all_ordercnt_day720=5,
                ),
                _build_source_row(),
            ]
        )

        result = build_verify_image_variables(df)

        expected_columns = [
            VERIFY_IMAGE_FIRST_BIND_MONTHS_COLUMN,
            VERIFY_IMAGE_LAST_BIND_MONTHS_COLUMN,
            *[
                build_verify_last_bind_months_feature_name("all", "all", "membercnt", source_group=source_group)
                for source_group in ("bank", "cf", "top", "others", "nonloan")
            ],
            *[
                build_verify_first_bind_months_feature_name("all", "all", "membercnt", source_group=source_group)
                for source_group in ("bank", "cf", "top", "others", "nonloan")
            ],
            *[
                build_verify_feature_name("all", "all", "all", metric, day_window)
                for metric in ("ordercnt", "membercnt")
                for day_window in (31, 60, 90, 180, 360, 720)
            ],
            *[
                build_verify_feature_name("all", source_group, "all", metric, day_window)
                for source_group in ("bank", "cf", "top", "others", "nonloan")
                for metric in ("ordercnt", "membercnt")
                for day_window in (31, 60, 90, 180, 360, 720)
            ],
        ]

        self.assertEqual(result.columns.tolist(), expected_columns)
        self.assertEqual(result.shape, (2, 84))
        self.assertEqual(result.loc[0, "verify_all_all_all_ordercnt_day31"], 4)
        self.assertEqual(result.loc[0, "verify_all_all_all_ordercnt_day720"], 5)
        self.assertEqual(result.loc[0, "verify_all_all_all_membercnt_day31"], 1)
        self.assertEqual(result.loc[0, "verify_all_all_all_membercnt_day60"], 2)
        self.assertEqual(result.loc[0, "verify_all_all_all_membercnt_day180"], 3)
        self.assertEqual(result.loc[0, VERIFY_IMAGE_FIRST_BIND_MONTHS_COLUMN], 6.0)
        self.assertEqual(result.loc[0, VERIFY_IMAGE_LAST_BIND_MONTHS_COLUMN], 1.0)
        self.assertEqual(result.loc[0, "verify_all_bank_all_membercnt_first_bind_months"], 1.0)
        self.assertEqual(result.loc[0, "verify_all_bank_all_membercnt_last_bind_months"], 1.0)
        self.assertEqual(result.loc[0, "verify_all_cf_all_membercnt_first_bind_months"], 2.0)
        self.assertEqual(result.loc[0, "verify_all_cf_all_membercnt_last_bind_months"], 2.0)
        self.assertEqual(result.loc[0, "verify_all_top_all_membercnt_first_bind_months"], 6.0)
        self.assertEqual(result.loc[0, "verify_all_top_all_membercnt_last_bind_months"], 6.0)
        self.assertTrue(pd.isna(result.loc[0, "verify_all_others_all_membercnt_first_bind_months"]))
        self.assertTrue(pd.isna(result.loc[0, "verify_all_others_all_membercnt_last_bind_months"]))
        self.assertTrue(pd.isna(result.loc[0, "verify_all_nonloan_all_membercnt_first_bind_months"]))
        self.assertTrue(pd.isna(result.loc[0, "verify_all_nonloan_all_membercnt_last_bind_months"]))
        self.assertTrue(pd.isna(result.loc[1, VERIFY_IMAGE_FIRST_BIND_MONTHS_COLUMN]))
        self.assertTrue(pd.isna(result.loc[1, VERIFY_IMAGE_LAST_BIND_MONTHS_COLUMN]))

    def test_transform_verify_features_can_preserve_original_columns(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "extra_col": "keep",
                    **_build_source_row(verify_all_bank_all_membercnt_day31=1),
                }
            ]
        )

        result = transform_verify_features(df, keep_only_target_columns=False)

        self.assertIn("extra_col", result.columns)
        self.assertEqual(result.loc[0, "extra_col"], "keep")
        self.assertEqual(result.loc[0, "verify_all_all_all_membercnt_day31"], 1)
        self.assertEqual(result.loc[0, VERIFY_IMAGE_LAST_BIND_MONTHS_COLUMN], 1.0)


if __name__ == "__main__":
    unittest.main()
