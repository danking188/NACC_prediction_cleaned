import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.ncaa_prediction.config import PipelineConfig
from src.ncaa_prediction.pipeline import run_stage2_pipeline


class EndToEndTests(unittest.TestCase):
    def test_synthetic_csv_to_trained_model_to_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for prefix, first in [("M", 1101), ("W", 3101)]:
                pd.DataFrame({"TeamID": [first, first + 1]}).to_csv(root / f"{prefix}Teams.csv", index=False)
                regular, tournament, seeds = [], [], []
                for season in [2023, 2024, 2025]:
                    for day in range(12):
                        winner = first + day % 2
                        loser = first + (1 - day % 2)
                        regular.append(dict(Season=season, DayNum=day + 1, WTeamID=winner, LTeamID=loser, WScore=80 + day, LScore=65 + day, WLoc="N"))
                        if season < 2025:
                            tournament.append(dict(Season=season, WTeamID=winner, LTeamID=loser))
                    seeds.extend([dict(Season=season, TeamID=first, Seed="W01"), dict(Season=season, TeamID=first + 1, Seed="W02")])
                pd.DataFrame(regular).to_csv(root / f"{prefix}RegularSeasonCompactResults.csv", index=False)
                pd.DataFrame(tournament).to_csv(root / f"{prefix}NCAATourneyCompactResults.csv", index=False)
                pd.DataFrame(seeds).to_csv(root / f"{prefix}NCAATourneySeeds.csv", index=False)
            # Keep a deliberately non-sorted gender order to check sample order preservation.
            ids = ["2025_3101_3102", "2025_1101_1102"]
            pd.DataFrame({"ID": ids, "Pred": [0.5, 0.5]}).to_csv(root / "SampleSubmissionStage2.csv", index=False)
            config = PipelineConfig(data_dir=root, output_path=root / "outputs/submission.csv", train_seasons=[2023], val_seasons=[2024], elo_start_season=2023, prediction_season=2025)
            result = run_stage2_pipeline(config)
            submission = pd.read_csv(config.output_path)
            self.assertEqual(submission.columns.tolist(), ["ID", "Pred"])
            self.assertEqual(submission["ID"].tolist(), ids)
            self.assertTrue(submission["Pred"].between(config.clip_min, config.clip_max).all())
            self.assertTrue(all(np.isfinite(v) for v in result["metrics"].values()))
            self.assertEqual(result["validation"]["missing"], 0)


if __name__ == "__main__":
    unittest.main()
