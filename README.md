# NCAA 2026 Stage 2 Prediction

This project generates predictions for the 2026 NCAA men's and women's tournament Stage 2 submission format.

这是一个 NCAA 篮球比赛概率预测项目（仓库名保留为 `NACC_prediction_cleaned`）。输出 `Pred` 表示 ID 中 **TeamA 战胜 TeamB 的概率**，并非已验证的比赛结果或成绩保证。

The current main pipeline is based on the previous `stage2_generation.py` algorithm:

- conservative ELO ratings from regular-season results
- season-level team statistics from compact and detailed regular-season data
- seed and ELO difference features
- calibrated logistic regression
- gradient boosting
- averaged ensemble prediction

## Project Layout

```text
data/raw/                 local NCAA CSV input files, not committed
outputs/                  generated submissions
models/                   local model artifacts, not committed
src/ncaa_prediction/      reusable pipeline modules
tests/                    lightweight tests
stage2_generation.py      main CLI entry point
```

## Setup

Use Python **3.11 or 3.12**. Run all commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell.
`requirements.txt` defines supported dependency ranges; `requirements.lock` pins the reproducible test environment.

Place the Kaggle NCAA CSV files in `data/raw/` before running the pipeline.
The raw data files are intentionally excluded from git because one source file
is larger than GitHub's normal 100 MB file limit.

Required files (both `M` and `W` prefixes):

```text
MTeams.csv                         WTeams.csv
MRegularSeasonCompactResults.csv   WRegularSeasonCompactResults.csv
MNCAATourneyCompactResults.csv     WNCAATourneyCompactResults.csv
MNCAATourneySeeds.csv              WNCAATourneySeeds.csv
SampleSubmissionStage2.csv
```

Optional: `MRegularSeasonDetailedResults.csv` and `WRegularSeasonDetailedResults.csv`.
Use the original Kaggle column names. Compact regular-season results need `Season`, `DayNum`, `WTeamID`, `LTeamID`, `WScore`, `LScore`, `WLoc`; teams need `TeamID`; seeds need `Season`, `TeamID`, `Seed`; tournament results need `Season`, `WTeamID`, `LTeamID`.
Detailed results, when supplied, must include `Season`, `WTeamID`, `LTeamID`, and both winner/loser columns for `FGM`, `FGM3`, `FGA`, `FTM`, `FTA`. Effective FG% is `(FGM + 0.5 × FGM3) / FGA`. Without detailed results, shooting features default to zero.

## Generate Stage 2 Submission

```bash
python stage2_generation.py
```

Default output:

```text
outputs/submission_stage2.csv
```

Custom paths:

```bash
python stage2_generation.py --data-dir data/raw --output outputs/my_submission.csv
```

## Validation

Run lightweight checks:

```bash
python -m unittest discover -s tests
```

The pipeline validates that generated IDs exactly match `SampleSubmissionStage2.csv` before saving output.

Validation also rejects duplicate/malformed IDs, reversed or identical teams, mixed-gender/unknown pairs, non-finite/out-of-range probabilities and reordered rows. Sample IDs must use `YYYY_TeamA_TeamB` with four digits per field, `TeamA < TeamB`, and match `--prediction-season`.

The test suite contains 14 tests, including a **synthetic CSV → real model training → submission file** integration test. No Kaggle download or credentials are needed for these tests. GitHub Actions runs tests and the CLI help check on Python 3.11/3.12.

## Modeling boundaries

- Default split: training 2015–2024, validation 2025, prediction 2026. Changing `--prediction-season` updates the default validation year to the previous year and ends training before validation. For custom season lists, instantiate `PipelineConfig` in Python.
- Training, validation and prediction must be chronologically ordered; ELO history must cover training. Calibration requires at least five training rows for each binary outcome.
- The validation season is used for evaluation, **not refitted into the final ensemble**. Calibration uses five-fold classification CV, not a time-series CV estimator.
- Raw CSV files must contain only information available before the target tournament. These checks do not independently establish the provenance or publication time of user-provided data.
- Synthetic tests establish pipeline behavior, not predictive quality. Real Kaggle data is not included, so no current leaderboard score or real-data validation metric is claimed.
- Re-running overwrites the chosen submission CSV. Keep important results under distinct output names.

## Notes

- The old exploratory `ncaa_*.py` scripts were removed to keep one maintainable pipeline.
- `outputs/` and `models/` are empty placeholders in a fresh clone; generated submissions and local models are ignored by Git.
- The current pipeline retrains from source CSV data and does not automatically persist the fitted model.
