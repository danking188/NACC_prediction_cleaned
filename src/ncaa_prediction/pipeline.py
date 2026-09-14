from typing import Dict, List, Tuple

import pandas as pd

from .config import PipelineConfig
from .data_loader import load_ncaa_data, validate_submission_shape
from .features import compute_elo_conservative, create_prediction_data, create_training_data
from .models import evaluate_model, train_ensemble


def run_stage2_pipeline(config: PipelineConfig) -> Dict:
    data = load_ncaa_data(config.data_dir)
    if set(data.sample_stage2["Season"]) != {config.prediction_season}:
        raise ValueError("Sample submission seasons must match prediction_season")
    # Reject invalid cross-gender/unknown pairs before expensive model fitting.
    _split_pairs_by_gender(data.sample_stage2, set(data.men.teams["TeamID"]), set(data.women.teams["TeamID"]))

    print("Computing ELO ratings...")
    men_elo = compute_elo_conservative(
        data.men.regular, config.elo_start_season, config.prediction_season
    )
    women_elo = compute_elo_conservative(
        data.women.regular, config.elo_start_season, config.prediction_season
    )

    print("Creating training and validation data...")
    men_train = create_training_data(
        data.men.tourney, data.men.regular, data.men.detailed,
        data.men.teams, data.men.seeds, config.train_seasons, men_elo
    )
    women_train = create_training_data(
        data.women.tourney, data.women.regular, data.women.detailed,
        data.women.teams, data.women.seeds, config.train_seasons, women_elo
    )
    men_val = create_training_data(
        data.men.tourney, data.men.regular, data.men.detailed,
        data.men.teams, data.men.seeds, config.val_seasons, men_elo
    )
    women_val = create_training_data(
        data.women.tourney, data.women.regular, data.women.detailed,
        data.women.teams, data.women.seeds, config.val_seasons, women_elo
    )

    train_df = pd.concat([men_train, women_train], ignore_index=True)
    val_df = pd.concat([men_val, women_val], ignore_index=True)
    if val_df.empty:
        raise ValueError("No tournament validation games found for the selected seasons")
    print(f"Training games: {len(train_df)}, validation games: {len(val_df)}")

    print("Training ensemble model...")
    model = train_ensemble(
        train_df,
        clip_min=config.clip_min,
        clip_max=config.clip_max,
        random_state=config.random_state,
    )
    metrics = evaluate_model(model, val_df)
    print(
        "Validation: "
        f"LogLoss={metrics['log_loss']:.4f}, "
        f"Brier={metrics['brier']:.4f}, "
        f"Accuracy={metrics['accuracy']:.4f}"
    )

    print("Generating Stage 2 predictions...")
    predictions = _predict_submission(data.sample_stage2, data, model, men_elo, women_elo)
    validation = validate_submission_shape(data.sample_stage2, predictions)
    if validation["missing"] or validation["extra"]:
        raise ValueError(f"Submission ID mismatch: {validation}")

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(config.output_path, index=False)
    print(f"Saved submission to {config.output_path}")
    print(
        "Prediction stats: "
        f"rows={len(predictions)}, "
        f"mean={predictions['Pred'].mean():.4f}, "
        f"std={predictions['Pred'].std():.4f}, "
        f"min={predictions['Pred'].min():.4f}, "
        f"max={predictions['Pred'].max():.4f}"
    )

    return {
        "metrics": metrics,
        "submission": str(config.output_path),
        "validation": validation,
    }


def _predict_submission(sample: pd.DataFrame, data, model, men_elo, women_elo) -> pd.DataFrame:
    men_ids = set(data.men.teams["TeamID"])
    women_ids = set(data.women.teams["TeamID"])
    pred_by_id: Dict[str, float] = {}

    for season in sorted(sample["Season"].unique()):
        season_sample = sample[sample["Season"] == season]
        men_pairs, women_pairs = _split_pairs_by_gender(season_sample, men_ids, women_ids)

        if men_pairs:
            men_pred_frame = create_prediction_data(
                men_pairs, season, data.men.regular, data.men.detailed,
                data.men.teams, data.men.seeds, men_elo
            )
            men_probs = model.predict_proba(men_pred_frame)
            for row, prob in zip(men_pred_frame.itertuples(index=False), men_probs):
                pred_by_id[f"{season}_{row.TeamA}_{row.TeamB}"] = float(prob)

        if women_pairs:
            women_pred_frame = create_prediction_data(
                women_pairs, season, data.women.regular, data.women.detailed,
                data.women.teams, data.women.seeds, women_elo
            )
            women_probs = model.predict_proba(women_pred_frame)
            for row, prob in zip(women_pred_frame.itertuples(index=False), women_probs):
                pred_by_id[f"{season}_{row.TeamA}_{row.TeamB}"] = float(prob)

    output = sample[["ID"]].copy()
    output["Pred"] = output["ID"].map(pred_by_id)
    if output["Pred"].isna().any():
        missing = output[output["Pred"].isna()]["ID"].head(10).tolist()
        raise ValueError(f"Missing predictions for IDs: {missing}")
    return output


def _split_pairs_by_gender(sample: pd.DataFrame, men_ids: set, women_ids: set) -> Tuple[List[tuple], List[tuple]]:
    men_pairs = []
    women_pairs = []
    for row in sample.itertuples(index=False):
        pair = (row.TeamA, row.TeamB)
        if row.TeamA in men_ids and row.TeamB in men_ids:
            men_pairs.append(pair)
        elif row.TeamA in women_ids and row.TeamB in women_ids:
            women_pairs.append(pair)
        else:
            raise ValueError(f"Cannot infer gender for matchup: {row.ID}")
    return men_pairs, women_pairs
