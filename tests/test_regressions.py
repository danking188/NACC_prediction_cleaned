import unittest

import numpy as np
import pandas as pd

from src.ncaa_prediction.config import PipelineConfig
from src.ncaa_prediction.data_loader import parse_submission_ids, validate_submission_shape
from src.ncaa_prediction.features import compute_team_stats
from src.ncaa_prediction.models import train_ensemble, evaluate_model
from src.ncaa_prediction.pipeline import _split_pairs_by_gender


class RegressionTests(unittest.TestCase):
    def test_bad_submission_ids(self):
        for ids in [[None], [123], ["2026_1101_1102_extra"], ["2026_1102_1101"], ["2026_1101_1101"], ["2026_1101_1102"] * 2, []]:
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                parse_submission_ids(pd.DataFrame({"ID": ids}))

    def test_probabilities_must_be_finite_and_in_range(self):
        sample = pd.DataFrame({"ID": ["2026_1101_1102"]})
        for probability in [float("nan"), float("inf"), -0.1, 1.1, "bad"]:
            with self.subTest(probability=probability), self.assertRaises(ValueError):
                validate_submission_shape(sample, sample.assign(Pred=probability))

    def test_duplicate_predictions_and_order_rejected(self):
        sample = pd.DataFrame({"ID": ["2026_1101_1102", "2026_1101_1103"]})
        with self.assertRaises(ValueError):
            validate_submission_shape(sample, pd.concat([sample, sample]).assign(Pred=0.5))
        with self.assertRaises(ValueError):
            validate_submission_shape(sample, sample.iloc[::-1].assign(Pred=0.5))

    def test_mixed_gender_and_unknown_teams_rejected(self):
        for matchup in ["2026_1101_3101", "2026_1101_9999"]:
            with self.subTest(matchup=matchup), self.assertRaises(ValueError):
                _split_pairs_by_gender(parse_submission_ids(pd.DataFrame({"ID": [matchup]})), {1101, 1102}, {3101, 3102})

    def test_invalid_config(self):
        for kwargs in [dict(clip_min=-1), dict(clip_max=1), dict(clip_min=float("nan")), dict(train_seasons=[]), dict(train_seasons=[2025]), dict(elo_start_season=2020)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PipelineConfig(**kwargs)
        self.assertEqual(PipelineConfig(prediction_season=2027).val_seasons, [2026])

    def test_effective_field_goal_percentage_includes_three_pointers(self):
        games = pd.DataFrame(dict(Season=[2025], DayNum=[10], WTeamID=[1101], LTeamID=[1102], WScore=[70], LScore=[60], WLoc=["N"]))
        detailed = games.assign(WFGM=20, WFGA=50, WFGM3=10, WFTM=20, WFTA=25, LFGM=20, LFGA=50, LFGM3=5, LFTM=15, LFTA=20)
        stats = compute_team_stats(games, detailed, pd.DataFrame({"TeamID": [1101, 1102]}), 2025)
        self.assertAlmostEqual(stats.loc[1101, "efg_pct"], 0.5)
        self.assertAlmostEqual(stats.loc[1102, "efg_pct"], 0.45)

    def test_too_small_training_set_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, "five games"):
            train_ensemble(pd.DataFrame({"Result": [0, 1]}))

    def test_single_class_validation(self):
        class ConstantModel:
            def predict_proba(self, frame):
                return np.full(len(frame), 0.6)
        self.assertTrue(np.isfinite(evaluate_model(ConstantModel(), pd.DataFrame({"Result": [1, 1]}))["log_loss"]))


if __name__ == "__main__":
    unittest.main()
