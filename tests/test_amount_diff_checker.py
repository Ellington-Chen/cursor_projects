import unittest

import pandas as pd

from amount_diff_checker import (
    check_card_amount_diffs_different_institutions_refactored,
    sample_data_by_idcard_dateback,
)


class AmountDiffCheckerTests(unittest.TestCase):
    def test_sampling_is_reproducible_with_seed(self) -> None:
        df = pd.DataFrame(
            {
                "idcard": ["i1", "i1", "i2", "i2", "i3", "i3"],
                "dateBack": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
                "card": ["c1", "c1", "c2", "c2", "c3", "c3"],
                "source": ["s1", "s2", "s1", "s2", "s1", "s2"],
            }
        )

        sample_one = sample_data_by_idcard_dateback(df, "idcard", "dateBack", sample_rate=2 / 3, sample_seed=7)
        sample_two = sample_data_by_idcard_dateback(df, "idcard", "dateBack", sample_rate=2 / 3, sample_seed=7)
        sample_three = sample_data_by_idcard_dateback(df, "idcard", "dateBack", sample_rate=2 / 3, sample_seed=8)

        self.assertTrue(sample_one.equals(sample_two))
        self.assertFalse(sample_one.equals(sample_three))

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
