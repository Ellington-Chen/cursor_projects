import unittest

import pandas as pd

from amount_diff_checker import (
    check_card_amount_diffs_different_institutions_refactored,
    sample_data_by_idcard_dateback,
)


class AmountDiffCheckerTests(unittest.TestCase):
    def test_sampling_is_reproducible_with_seed(self) -> None:
        rows = []
        for index in range(1, 9):
            rows.append(
                {
                    "idcard": f"i{index}",
                    "dateBack": f"2024-01-{index:02d}",
                    "card": f"c{index}",
                    "source": "s1",
                }
            )
            rows.append(
                {
                    "idcard": f"i{index}",
                    "dateBack": f"2024-01-{index:02d}",
                    "card": f"c{index}",
                    "source": "s2",
                }
            )
        df = pd.DataFrame(rows)

        sample_one = sample_data_by_idcard_dateback(df, "idcard", "dateBack", sample_rate=0.5, sample_seed=7)
        sample_two = sample_data_by_idcard_dateback(df, "idcard", "dateBack", sample_rate=0.5, sample_seed=7)
        sample_three = sample_data_by_idcard_dateback(df, "idcard", "dateBack", sample_rate=0.5, sample_seed=8)

        sampled_keys_one = sorted(map(tuple, sample_one[["idcard", "dateBack"]].drop_duplicates().values.tolist()))
        sampled_keys_three = sorted(map(tuple, sample_three[["idcard", "dateBack"]].drop_duplicates().values.tolist()))
        expected_seed_seven_keys = [
            ("i1", "2024-01-01"),
            ("i3", "2024-01-03"),
            ("i6", "2024-01-06"),
            ("i7", "2024-01-07"),
        ]
        expected_seed_eight_keys = [
            ("i1", "2024-01-01"),
            ("i3", "2024-01-03"),
            ("i7", "2024-01-07"),
            ("i8", "2024-01-08"),
        ]

        self.assertTrue(sample_one.equals(sample_two))
        self.assertEqual(sampled_keys_one, expected_seed_seven_keys)
        self.assertEqual(sampled_keys_three, expected_seed_eight_keys)

    def test_cross_institution_match_is_detected(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "idcard": "id-1",
                    "dateBack": "2024-01-01",
                    "card": "card-1",
                    "source": "source-a",
                    "pre_1_bank_fail_out_max": '{"inst-b": 100.0}',
                },
                {
                    "idcard": "id-1",
                    "dateBack": "2024-01-01",
                    "card": "card-1",
                    "source": "source-b",
                    "pre_1_bank_fail_out_max": '{"inst-a": 100.2}',
                },
            ]
        )

        results, summary = check_card_amount_diffs_different_institutions_refactored(
            df,
            use_sampling=False,
            diff_threshold=0.5,
        )

        self.assertEqual(summary["total_results"], 1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results.iloc[0]["institution1"], "inst-b")
        self.assertEqual(results.iloc[0]["institution2"], "inst-a")

    def test_default_group_keys_do_not_mix_different_idcards(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "idcard": "id-1",
                    "dateBack": "2024-01-01",
                    "card": "card-1",
                    "source": "source-a",
                    "pre_1_bank_fail_out_max": '{"inst-a": 100.0}',
                },
                {
                    "idcard": "id-2",
                    "dateBack": "2024-01-01",
                    "card": "card-1",
                    "source": "source-b",
                    "pre_1_bank_fail_out_max": '{"inst-b": 100.2}',
                },
            ]
        )

        results, summary = check_card_amount_diffs_different_institutions_refactored(
            df,
            use_sampling=False,
            diff_threshold=0.5,
        )

        self.assertTrue(results.empty)
        self.assertEqual(summary["total_results"], 0)

    def test_parse_errors_are_counted(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "idcard": "id-1",
                    "dateBack": "2024-01-01",
                    "card": "card-1",
                    "source": "source-a",
                    "pre_1_bank_fail_out_max": '{"inst-a": 100.0}',
                },
                {
                    "idcard": "id-1",
                    "dateBack": "2024-01-01",
                    "card": "card-1",
                    "source": "source-b",
                    "pre_1_bank_fail_out_max": "{broken json",
                },
            ]
        )

        _, summary = check_card_amount_diffs_different_institutions_refactored(
            df,
            use_sampling=False,
        )

        self.assertEqual(summary["parse_error_count"], 1)


if __name__ == "__main__":
    unittest.main()
