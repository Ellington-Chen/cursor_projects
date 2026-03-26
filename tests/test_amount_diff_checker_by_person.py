import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

from amount_diff_checker_by_person import (
    check_person_amount_diffs_different_institutions_refactored,
    sample_data_by_idcard_dateback,
)


class AmountDiffCheckerByPersonTests(unittest.TestCase):
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

    def test_cross_institution_same_card_match(self) -> None:
        """同一人同一日期、同一卡号、不同 source/institution，应命中。"""
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

        results, summary = check_person_amount_diffs_different_institutions_refactored(
            df,
            use_sampling=False,
            diff_threshold=0.5,
        )

        self.assertEqual(summary["total_results"], 1)
        self.assertEqual(summary["total_candidate_group_count"], 1)
        self.assertEqual(summary["total_unique_idcards"], 1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results.iloc[0]["card1"], "card-1")
        self.assertEqual(results.iloc[0]["card2"], "card-1")
        self.assertEqual(results.iloc[0]["institution1"], "inst-b")
        self.assertEqual(results.iloc[0]["institution2"], "inst-a")

    def test_cross_institution_different_cards_match(self) -> None:
        """同一人同一日期、不同卡号，不同 source/institution 的金额接近，应命中。"""
        df = pd.DataFrame(
            [
                {
                    "idcard": "id-1",
                    "dateBack": "2024-01-01",
                    "card": "card-A",
                    "source": "source-a",
                    "pre_1_bank_fail_out_max": '{"inst-x": 200.0}',
                },
                {
                    "idcard": "id-1",
                    "dateBack": "2024-01-01",
                    "card": "card-B",
                    "source": "source-b",
                    "pre_1_bank_fail_out_max": '{"inst-y": 200.3}',
                },
            ]
        )

        results, summary = check_person_amount_diffs_different_institutions_refactored(
            df,
            use_sampling=False,
            diff_threshold=0.5,
        )

        self.assertEqual(summary["total_results"], 1)
        self.assertEqual(len(results), 1)
        row = results.iloc[0]
        self.assertEqual(row["card1"], "card-A")
        self.assertEqual(row["card2"], "card-B")
        self.assertAlmostEqual(row["diff"], 0.3, places=5)

    def test_different_idcards_do_not_mix(self) -> None:
        """不同身份证不会被分到同一组。"""
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

        results, summary = check_person_amount_diffs_different_institutions_refactored(
            df,
            use_sampling=False,
            diff_threshold=0.5,
        )

        self.assertTrue(results.empty)
        self.assertEqual(summary["total_results"], 0)

    def test_default_group_keys_is_idcard_dateback(self) -> None:
        """默认 group_keys 应该是 [idcard, dateBack]。"""
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
                    "pre_1_bank_fail_out_max": '{"inst-b": 100.2}',
                },
            ]
        )

        _, summary = check_person_amount_diffs_different_institutions_refactored(
            df,
            use_sampling=False,
            diff_threshold=0.5,
            show_progress=False,
        )

        self.assertEqual(summary["group_keys"], ["idcard", "dateBack"])

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

        _, summary = check_person_amount_diffs_different_institutions_refactored(
            df,
            use_sampling=False,
        )

        self.assertEqual(summary["parse_error_count"], 1)

    def test_unique_cards_csv_contains_card1_and_card2(self) -> None:
        """CSV 的 card 列应包含 card1 和 card2 的去重合集。"""
        df = pd.DataFrame(
            [
                {
                    "idcard": "id-1",
                    "dateBack": "2024-01-01",
                    "card": "card-A",
                    "source": "source-a",
                    "pre_1_bank_fail_out_max": '{"inst-b": 100.0}',
                },
                {
                    "idcard": "id-1",
                    "dateBack": "2024-01-01",
                    "card": "card-B",
                    "source": "source-b",
                    "pre_1_bank_fail_out_max": '{"inst-a": 100.2}',
                },
                {
                    "idcard": "id-2",
                    "dateBack": "2024-01-02",
                    "card": "card-C",
                    "source": "source-a",
                    "pre_1_bank_fail_out_max": '{"inst-d": 200.0}',
                },
                {
                    "idcard": "id-2",
                    "dateBack": "2024-01-02",
                    "card": "card-C",
                    "source": "source-b",
                    "pre_1_bank_fail_out_max": '{"inst-c": 200.3}',
                },
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "unique_cards.csv"
            results, summary = check_person_amount_diffs_different_institutions_refactored(
                df,
                use_sampling=False,
                diff_threshold=0.5,
                unique_cards_output_path=output_path,
                show_progress=False,
            )

            self.assertEqual(len(results), 2)
            self.assertNotIn("unique_cards", summary)
            self.assertEqual(summary["total_unique_cards"], 3)
            self.assertEqual(summary["total_unique_idcards"], 2)
            self.assertTrue(output_path.exists())

            unique_cards_df = pd.read_csv(output_path)
            self.assertEqual(
                unique_cards_df["card"].tolist(),
                ["card-A", "card-B", "card-C"],
            )
            self.assertIn("pre_1_bank_fail_out_max", unique_cards_df.columns)

    def test_multi_amount_cols_csv_has_per_column_cards(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "idcard": "id-1",
                    "dateBack": "2024-01-01",
                    "card": "card-1",
                    "source": "source-a",
                    "col_a": '{"inst-x": 10.0}',
                    "col_b": '{"inst-y": 20.0}',
                },
                {
                    "idcard": "id-1",
                    "dateBack": "2024-01-01",
                    "card": "card-2",
                    "source": "source-b",
                    "col_a": '{"inst-z": 10.3}',
                    "col_b": '{"inst-w": 99.0}',
                },
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "unique_cards.csv"
            results, summary = check_person_amount_diffs_different_institutions_refactored(
                df,
                use_sampling=False,
                diff_threshold=0.5,
                amount_json_cols=["col_a", "col_b"],
                unique_cards_output_path=output_path,
                show_progress=False,
            )

            self.assertIn("total_candidate_group_count", summary)
            self.assertNotIn("cards", summary["column_stats"]["col_a"])
            self.assertNotIn("cards", summary["column_stats"]["col_b"])

            unique_cards_df = pd.read_csv(output_path)
            self.assertIn("card", unique_cards_df.columns)
            self.assertIn("col_a", unique_cards_df.columns)
            self.assertIn("col_b", unique_cards_df.columns)

            self.assertEqual(
                unique_cards_df["col_a"].dropna().tolist(),
                sorted(set(unique_cards_df["col_a"].dropna().tolist())),
            )

    def test_text_progress_fallback_prints_when_tqdm_is_unavailable(self) -> None:
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

        from amount_diff_checker_by_person import TQDM_AVAILABLE
        import amount_diff_checker_by_person

        previous_value = TQDM_AVAILABLE
        amount_diff_checker_by_person.TQDM_AVAILABLE = False
        buffer = io.StringIO()
        try:
            with redirect_stdout(buffer):
                check_person_amount_diffs_different_institutions_refactored(
                    df,
                    use_sampling=False,
                    diff_threshold=0.5,
                    show_progress=True,
                    unique_cards_output_path=None,
                )
        finally:
            amount_diff_checker_by_person.TQDM_AVAILABLE = previous_value

        progress_output = buffer.getvalue()
        self.assertIn("处理金额列", progress_output)
        self.assertIn("解析 pre_1_bank_fail_out_max", progress_output)
        self.assertIn("匹配 pre_1_bank_fail_out_max", progress_output)


if __name__ == "__main__":
    unittest.main()
