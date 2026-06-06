from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd


@dataclass
class GenderData:
    teams: pd.DataFrame
    regular: pd.DataFrame
    detailed: pd.DataFrame
    tourney: pd.DataFrame
    seeds: pd.DataFrame


@dataclass
class NCAAData:
    men: GenderData
    women: GenderData
    sample_stage2: pd.DataFrame


def _read_csv(data_dir: Path, name: str) -> pd.DataFrame:
    path = data_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    return pd.read_csv(path)


def _read_optional_csv(data_dir: Path, name: str) -> pd.DataFrame:
    path = data_dir / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_gender_data(data_dir: Path, prefix: str) -> GenderData:
    return GenderData(
        teams=_read_csv(data_dir, f"{prefix}Teams.csv"),
        regular=_read_csv(data_dir, f"{prefix}RegularSeasonCompactResults.csv"),
        detailed=_read_optional_csv(data_dir, f"{prefix}RegularSeasonDetailedResults.csv"),
        tourney=_read_csv(data_dir, f"{prefix}NCAATourneyCompactResults.csv"),
        seeds=_read_csv(data_dir, f"{prefix}NCAATourneySeeds.csv"),
    )


def load_ncaa_data(data_dir: Path) -> NCAAData:
    data_dir = Path(data_dir)
    sample = _read_csv(data_dir, "SampleSubmissionStage2.csv")
    sample = parse_submission_ids(sample)
    return NCAAData(
        men=load_gender_data(data_dir, "M"),
        women=load_gender_data(data_dir, "W"),
        sample_stage2=sample,
    )


def parse_submission_ids(submission: pd.DataFrame) -> pd.DataFrame:
    parsed = submission.copy()
    parts = parsed["ID"].str.split("_", expand=True)
    parsed["Season"] = parts[0].astype(int)
    parsed["TeamA"] = parts[1].astype(int)
    parsed["TeamB"] = parts[2].astype(int)
    return parsed


def validate_submission_shape(sample: pd.DataFrame, predictions: pd.DataFrame) -> Dict[str, int]:
    required_ids = set(sample["ID"])
    generated_ids = set(predictions["ID"])
    return {
        "required": len(required_ids),
        "generated": len(generated_ids),
        "missing": len(required_ids - generated_ids),
        "extra": len(generated_ids - required_ids),
    }
