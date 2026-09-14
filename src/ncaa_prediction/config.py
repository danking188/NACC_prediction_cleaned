from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class PipelineConfig:
    data_dir: Path = Path("data/raw")
    output_path: Path = Path("outputs/submission_stage2.csv")
    train_seasons: List[int] = None
    val_seasons: List[int] = None
    elo_start_season: int = 2015
    prediction_season: int = 2026
    clip_min: float = 0.001
    clip_max: float = 0.999
    random_state: int = 42

    def __post_init__(self):
        if self.train_seasons is None:
            self.train_seasons = list(range(2015, self.prediction_season - 1))
        if self.val_seasons is None:
            self.val_seasons = [self.prediction_season - 1]
        if not self.train_seasons or not self.val_seasons:
            raise ValueError("Training and validation seasons must be non-empty")
        if max(self.train_seasons) >= min(self.val_seasons) or max(self.val_seasons) >= self.prediction_season:
            raise ValueError("Training seasons must precede validation seasons, which must precede prediction")
        if self.elo_start_season > min(self.train_seasons):
            raise ValueError("ELO history must begin no later than the first training season")
        if not 0 < self.clip_min < self.clip_max < 1:
            raise ValueError("Probability bounds must satisfy 0 < clip_min < clip_max < 1")
        self.data_dir = Path(self.data_dir)
        self.output_path = Path(self.output_path)
