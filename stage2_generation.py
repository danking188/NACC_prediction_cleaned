"""
Generate NCAA 2026 Stage 2 tournament predictions.

This is the main entry point. It uses the cleaned Stage 2 pipeline based on
the previous stage2_generation.py algorithm: conservative ELO, regular-season
team statistics, calibrated logistic regression, gradient boosting, and a
simple averaged ensemble.
"""

import argparse
from pathlib import Path

from src.ncaa_prediction.config import PipelineConfig
from src.ncaa_prediction.pipeline import run_stage2_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate NCAA Stage 2 predictions")
    parser.add_argument("--data-dir", default="data/raw", help="Directory containing NCAA CSV files")
    parser.add_argument("--output", default="outputs/submission_stage2.csv", help="Submission output path")
    parser.add_argument("--elo-start-season", type=int, default=2015, help="First season used for ELO history")
    parser.add_argument("--prediction-season", type=int, default=2026, help="Season to predict")
    parser.add_argument("--clip-min", type=float, default=0.001, help="Minimum prediction probability")
    parser.add_argument("--clip-max", type=float, default=0.999, help="Maximum prediction probability")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(
        data_dir=Path(args.data_dir),
        output_path=Path(args.output),
        elo_start_season=args.elo_start_season,
        prediction_season=args.prediction_season,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
        random_state=args.random_state,
    )
    run_stage2_pipeline(config)


if __name__ == "__main__":
    main()
