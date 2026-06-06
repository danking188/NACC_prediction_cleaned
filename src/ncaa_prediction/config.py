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
            self.train_seasons = list(range(2015, 2025))
        if self.val_seasons is None:
            self.val_seasons = [2025]
        self.data_dir = Path(self.data_dir)
        self.output_path = Path(self.output_path)
