from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "win_pct_diff",
    "pts_pg_diff",
    "pts_allowed_pg_diff",
    "pt_diff_pg_diff",
    "home_win_pct_diff",
    "away_win_pct_diff",
    "sos_diff",
    "seed_diff",
    "elo_diff",
    "efg_pct_diff",
    "ftp_pct_diff",
]


def compute_elo_conservative(reg_df: pd.DataFrame, start_season: int = 2015,
                             end_season: int = 2026) -> Dict[int, Dict[int, float]]:
    """Compute season-end ELO ratings using only regular-season games."""
    base_elo = 1500
    k_factor = 15
    regression = 0.65

    all_teams = set(reg_df["WTeamID"].unique()) | set(reg_df["LTeamID"].unique())
    ratings = {team_id: base_elo for team_id in all_teams}
    season_ratings = {}

    for season in range(start_season, end_season + 1):
        if season > start_season:
            ratings = {
                team_id: base_elo * (1 - regression) + ratings.get(team_id, base_elo) * regression
                for team_id in all_teams
            }

        season_games = reg_df[reg_df["Season"] == season].sort_values("DayNum")
        for _, game in season_games.iterrows():
            winner = game["WTeamID"]
            loser = game["LTeamID"]
            if winner not in ratings or loser not in ratings:
                continue

            winner_elo = ratings[winner]
            loser_elo = ratings[loser]
            winner_expected = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
            loser_expected = 1 - winner_expected

            ratings[winner] = winner_elo + k_factor * (1 - winner_expected)
            ratings[loser] = loser_elo + k_factor * (0 - loser_expected)

        season_ratings[season] = ratings.copy()

    return season_ratings


def _empty_team_stats() -> Dict[str, float]:
    return {
        "wins": 0,
        "losses": 0,
        "pts_scored": 0,
        "pts_allowed": 0,
        "point_diff": 0,
        "home_wins": 0,
        "away_wins": 0,
        "home_losses": 0,
        "away_losses": 0,
        "fgm": 0,
        "fga": 0,
        "ftm": 0,
        "fta": 0,
    }


def compute_team_stats(reg_df: pd.DataFrame, det_df: pd.DataFrame,
                       teams_df: pd.DataFrame, season: int) -> pd.DataFrame:
    """Compute pre-tournament team stats for one season."""
    stats = {team_id: _empty_team_stats() for team_id in teams_df["TeamID"]}
    season_games = reg_df[reg_df["Season"] == season].sort_values("DayNum")

    for _, game in season_games.iterrows():
        winner = game["WTeamID"]
        loser = game["LTeamID"]
        winner_score = game["WScore"]
        loser_score = game["LScore"]
        location = game["WLoc"]

        stats.setdefault(winner, _empty_team_stats())
        stats.setdefault(loser, _empty_team_stats())

        stats[winner]["wins"] += 1
        stats[loser]["losses"] += 1
        stats[winner]["pts_scored"] += winner_score
        stats[loser]["pts_scored"] += loser_score
        stats[winner]["pts_allowed"] += loser_score
        stats[loser]["pts_allowed"] += winner_score
        stats[winner]["point_diff"] += winner_score - loser_score
        stats[loser]["point_diff"] -= winner_score - loser_score

        if location == "H":
            stats[winner]["home_wins"] += 1
            stats[loser]["away_losses"] += 1
        elif location == "A":
            stats[winner]["away_wins"] += 1
            stats[loser]["home_losses"] += 1

    has_detailed_columns = det_df is not None and "Season" in det_df.columns
    season_det = det_df[det_df["Season"] == season] if has_detailed_columns else None
    if season_det is not None and not season_det.empty:
        for team_id in stats:
            wins = season_det[season_det["WTeamID"] == team_id]
            losses = season_det[season_det["LTeamID"] == team_id]
            if len(wins) + len(losses) == 0:
                continue

            stats[team_id]["fgm"] = wins["WFGM"].sum() + losses["LFGM"].sum()
            stats[team_id]["fga"] = wins["WFGA"].sum() + losses["LFGA"].sum()
            stats[team_id]["ftm"] = wins["WFTM"].sum() + losses["LFTM"].sum()
            stats[team_id]["fta"] = wins["WFTA"].sum() + losses["LFTA"].sum()

    stats_df = pd.DataFrame.from_dict(stats, orient="index")
    stats_df.index.name = "TeamID"
    stats_df["games"] = stats_df["wins"] + stats_df["losses"]
    safe_games = stats_df["games"].replace(0, 1)
    stats_df["win_pct"] = stats_df["wins"] / safe_games
    stats_df["pts_pg"] = stats_df["pts_scored"] / safe_games
    stats_df["pts_allowed_pg"] = stats_df["pts_allowed"] / safe_games
    stats_df["pt_diff_pg"] = stats_df["point_diff"] / safe_games
    stats_df["home_win_pct"] = stats_df["home_wins"] / (
        stats_df["home_wins"] + stats_df["home_losses"]
    ).replace(0, 1)
    stats_df["away_win_pct"] = stats_df["away_wins"] / (
        stats_df["away_wins"] + stats_df["away_losses"]
    ).replace(0, 1)
    stats_df["efg_pct"] = stats_df["fgm"] / stats_df["fga"].replace(0, 1)
    stats_df["ftp_pct"] = stats_df["ftm"] / stats_df["fta"].replace(0, 1)
    stats_df["sos"] = _compute_strength_of_schedule(season_games, stats_df)
    return stats_df


def _compute_strength_of_schedule(season_games: pd.DataFrame, stats_df: pd.DataFrame) -> pd.Series:
    sos = {}
    for team_id in stats_df.index:
        team_games = season_games[
            (season_games["WTeamID"] == team_id) | (season_games["LTeamID"] == team_id)
        ]
        if team_games.empty:
            sos[team_id] = 0.5
            continue

        opponent_wins = 0
        opponent_games = 0
        for _, game in team_games.iterrows():
            opponent = game["LTeamID"] if game["WTeamID"] == team_id else game["WTeamID"]
            if opponent in stats_df.index:
                opponent_wins += stats_df.loc[opponent, "wins"]
                opponent_games += stats_df.loc[opponent, "games"]
        sos[team_id] = opponent_wins / opponent_games if opponent_games > 0 else 0.5
    return pd.Series(sos)


def get_seed(seeds_df: pd.DataFrame, season: int, team_id: int, default_seed: int = 16) -> int:
    seeds = seeds_df[(seeds_df["Season"] == season) & (seeds_df["TeamID"] == team_id)]
    if seeds.empty:
        return default_seed
    return int(str(seeds.iloc[0]["Seed"])[1:3])


def get_team_stats_safe(stats_df: pd.DataFrame, team_id: int, default_val: float = 0.5) -> pd.Series:
    if team_id in stats_df.index:
        return stats_df.loc[team_id]
    return pd.Series({
        "win_pct": default_val,
        "pts_pg": 70,
        "pts_allowed_pg": 70,
        "pt_diff_pg": 0,
        "home_win_pct": default_val,
        "away_win_pct": default_val,
        "sos": default_val,
        "efg_pct": default_val,
        "ftp_pct": 0.7,
    })


def make_matchup_features(team_a: int, team_b: int, season: int, team_stats: pd.DataFrame,
                          seeds_df: pd.DataFrame, season_elo: Dict[int, float],
                          result: int = None) -> Dict:
    a_stats = get_team_stats_safe(team_stats, team_a)
    b_stats = get_team_stats_safe(team_stats, team_b)
    a_seed = get_seed(seeds_df, season, team_a)
    b_seed = get_seed(seeds_df, season, team_b)

    features = {
        "TeamA": team_a,
        "TeamB": team_b,
        "Season": season,
        "A_win_pct": a_stats["win_pct"],
        "B_win_pct": b_stats["win_pct"],
        "A_pts_pg": a_stats["pts_pg"],
        "B_pts_pg": b_stats["pts_pg"],
        "A_pts_allowed_pg": a_stats["pts_allowed_pg"],
        "B_pts_allowed_pg": b_stats["pts_allowed_pg"],
        "A_pt_diff_pg": a_stats["pt_diff_pg"],
        "B_pt_diff_pg": b_stats["pt_diff_pg"],
        "A_home_win_pct": a_stats["home_win_pct"],
        "B_home_win_pct": b_stats["home_win_pct"],
        "A_away_win_pct": a_stats["away_win_pct"],
        "B_away_win_pct": b_stats["away_win_pct"],
        "A_sos": a_stats["sos"],
        "B_sos": b_stats["sos"],
        "A_efg_pct": a_stats["efg_pct"],
        "B_efg_pct": b_stats["efg_pct"],
        "A_ftp_pct": a_stats["ftp_pct"],
        "B_ftp_pct": b_stats["ftp_pct"],
        "A_seed": a_seed,
        "B_seed": b_seed,
        "A_elo": season_elo.get(team_a, 1500),
        "B_elo": season_elo.get(team_b, 1500),
    }

    if result is not None:
        features["Result"] = result

    features["win_pct_diff"] = features["A_win_pct"] - features["B_win_pct"]
    features["pts_pg_diff"] = features["A_pts_pg"] - features["B_pts_pg"]
    features["pts_allowed_pg_diff"] = features["A_pts_allowed_pg"] - features["B_pts_allowed_pg"]
    features["pt_diff_pg_diff"] = features["A_pt_diff_pg"] - features["B_pt_diff_pg"]
    features["home_win_pct_diff"] = features["A_home_win_pct"] - features["B_home_win_pct"]
    features["away_win_pct_diff"] = features["A_away_win_pct"] - features["B_away_win_pct"]
    features["sos_diff"] = features["A_sos"] - features["B_sos"]
    features["seed_diff"] = features["A_seed"] - features["B_seed"]
    features["elo_diff"] = features["A_elo"] - features["B_elo"]
    features["efg_pct_diff"] = features["A_efg_pct"] - features["B_efg_pct"]
    features["ftp_pct_diff"] = features["A_ftp_pct"] - features["B_ftp_pct"]
    return features


def create_training_data(tourney_df: pd.DataFrame, reg_df: pd.DataFrame, det_df: pd.DataFrame,
                         teams_df: pd.DataFrame, seeds_df: pd.DataFrame,
                         seasons: Iterable[int], elo_ratings: Dict[int, Dict[int, float]]) -> pd.DataFrame:
    rows: List[Dict] = []
    for season in seasons:
        tourney = tourney_df[tourney_df["Season"] == season]
        team_stats = compute_team_stats(reg_df, det_df, teams_df, season)
        season_elo = elo_ratings.get(season, {})

        for _, game in tourney.iterrows():
            winner = game["WTeamID"]
            loser = game["LTeamID"]
            team_a = min(winner, loser)
            team_b = max(winner, loser)
            result = 1 if winner < loser else 0
            rows.append(make_matchup_features(
                team_a, team_b, season, team_stats, seeds_df, season_elo, result
            ))
    return pd.DataFrame(rows)


def create_prediction_data(pairs: Iterable[tuple], season: int, reg_df: pd.DataFrame,
                           det_df: pd.DataFrame, teams_df: pd.DataFrame,
                           seeds_df: pd.DataFrame, elo_ratings: Dict[int, Dict[int, float]]) -> pd.DataFrame:
    team_stats = compute_team_stats(reg_df, det_df, teams_df, season)
    season_elo = elo_ratings.get(season, {})
    rows = [
        make_matchup_features(team_a, team_b, season, team_stats, seeds_df, season_elo)
        for team_a, team_b in pairs
    ]
    return pd.DataFrame(rows)
