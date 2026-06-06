import unittest

import pandas as pd

from src.ncaa_prediction.features import compute_elo_conservative, compute_team_stats, get_seed


class FeatureTests(unittest.TestCase):
    def test_get_seed_defaults_to_16_when_missing(self):
        seeds = pd.DataFrame({
            "Season": [2026],
            "Seed": ["W01"],
            "TeamID": [1101],
        })

        self.assertEqual(get_seed(seeds, 2026, 1101), 1)
        self.assertEqual(get_seed(seeds, 2026, 9999), 16)

    def test_elo_returns_requested_season(self):
        games = pd.DataFrame({
            "Season": [2025],
            "DayNum": [10],
            "WTeamID": [1101],
            "LTeamID": [1102],
            "WScore": [80],
            "LScore": [70],
            "WLoc": ["N"],
            "NumOT": [0],
        })

        ratings = compute_elo_conservative(games, 2025, 2025)
        self.assertIn(2025, ratings)
        self.assertGreater(ratings[2025][1101], 1500)
        self.assertLess(ratings[2025][1102], 1500)

    def test_team_stats_tolerates_missing_detailed_results(self):
        games = pd.DataFrame({
            "Season": [2025],
            "DayNum": [10],
            "WTeamID": [1101],
            "LTeamID": [1102],
            "WScore": [80],
            "LScore": [70],
            "WLoc": ["N"],
            "NumOT": [0],
        })
        teams = pd.DataFrame({"TeamID": [1101, 1102]})

        stats = compute_team_stats(games, pd.DataFrame(), teams, 2025)

        self.assertEqual(stats.loc[1101, "win_pct"], 1.0)
        self.assertEqual(stats.loc[1102, "win_pct"], 0.0)
        self.assertEqual(stats.loc[1101, "efg_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
