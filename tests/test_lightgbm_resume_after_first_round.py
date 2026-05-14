import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import pandas as pd

import lightgbm_resume_after_first_round
from lightgbm_resume_after_first_round import (
    _train_lgb_model_compat,
    get_backward_selection_random_search_space,
    run_backward_selection_after_first_round,
)


class LightGBMResumeAfterFirstRoundTests(unittest.TestCase):
    def _build_frames(self) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        def build_frame(start: int, rows: int) -> pd.DataFrame:
            values = list(range(start, start + rows))
            return pd.DataFrame(
                {
                    "f1": [float(value) for value in values],
                    "f2": [float((value * 3) % 11) for value in values],
                    "unused": [float(value % 5) for value in values],
                }
            )

        X_train = build_frame(0, 120)
        y_train = pd.Series([index % 2 for index in range(120)])
        X_val = build_frame(1000, 60)
        y_val = pd.Series([index % 2 for index in range(60)])
        X_oot = build_frame(2000, 60)
        y_oot = pd.Series([index % 2 for index in range(60)])
        return X_train, y_train, X_val, y_val, X_oot, y_oot

    def test_random_search_space_matches_original_ranges(self) -> None:
        space = get_backward_selection_random_search_space()
        self.assertEqual(space["max_bin"], list(range(200, 2001, 100)))
        self.assertEqual(space["n_estimators"], list(range(20, 301, 20)))
        self.assertEqual(space["scale_pos_weight"], [0.8, 1])
        self.assertEqual(space["min_child_samples"], list(range(2000, 14001, 500)))

    def test_run_backward_selection_after_first_round_runs_from_random_search(self) -> None:
        X_train, y_train, X_val, y_val, X_oot, y_oot = self._build_frames()

        result = run_backward_selection_after_first_round(
            X_train,
            y_train,
            X_val,
            y_val,
            X_oot,
            y_oot,
            selected_features=["f1", "f2"],
            n_random_search_iter=1,
            random_search_cv=2,
            random_search_n_jobs=1,
            model_n_jobs=1,
            show_progress=False,
        )

        self.assertEqual(result.current_features, ["f1", "f2"])
        self.assertEqual(set(result.best_params.keys()), set(get_backward_selection_random_search_space().keys()))
        self.assertIn("auc", result.metricsTrain)
        self.assertIn("auc", result.metricsTest)
        self.assertIn("auc", result.metricsOOT)
        self.assertIn("ks", result.metricsTest)
        self.assertIn("lift", result.metricsOOT)
        self.assertFalse(result.feature_importance_df.empty)
        self.assertIsNotNone(result.model)
        self.assertIsNotNone(result.random_search)

    def test_run_backward_selection_after_first_round_prints_text_progress_without_tqdm(self) -> None:
        X_train, y_train, X_val, y_val, X_oot, y_oot = self._build_frames()
        previous_value = lightgbm_resume_after_first_round.TQDM_AVAILABLE
        lightgbm_resume_after_first_round.TQDM_AVAILABLE = False
        buffer = io.StringIO()

        try:
            with redirect_stdout(buffer):
                result = run_backward_selection_after_first_round(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    X_oot,
                    y_oot,
                    selected_features=["f1", "f2"],
                    n_random_search_iter=1,
                    random_search_cv=2,
                    random_search_n_jobs=1,
                    model_n_jobs=1,
                    show_progress=True,
                )
        finally:
            lightgbm_resume_after_first_round.TQDM_AVAILABLE = previous_value

        progress_output = buffer.getvalue()
        self.assertIsNotNone(result.model)
        self.assertIn("random_search_after_first_round: starting", progress_output)
        self.assertIn("best_score=", progress_output)

    def test_train_lgb_model_compat_falls_back_when_verbose_false_unsupported(self) -> None:
        class FakeLightGBM:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def early_stopping(self, stopping_rounds: int, verbose: bool = True) -> tuple[int, bool]:
                return (stopping_rounds, verbose)

            def train(self, params: dict[str, object], train_data: object, **kwargs: object) -> str:
                self.calls.append({"params": params, "kwargs": kwargs})
                callbacks = kwargs.get("callbacks")
                if callbacks == [(20, False)]:
                    raise TypeError("early_stopping(verbose=False) unsupported")
                return "trained_model"

        fake_lgb = FakeLightGBM()
        model = _train_lgb_model_compat(
            fake_lgb,
            {"objective": "binary", "verbose": -1},
            train_data=object(),
            valid_data=object(),
            num_boost_round=100,
            stopping_rounds=20,
        )

        self.assertEqual(model, "trained_model")
        self.assertEqual(len(fake_lgb.calls), 2)
        self.assertEqual(fake_lgb.calls[0]["kwargs"]["callbacks"], [(20, False)])
        self.assertEqual(fake_lgb.calls[1]["kwargs"]["callbacks"], [(20, True)])


if __name__ == "__main__":
    unittest.main()
