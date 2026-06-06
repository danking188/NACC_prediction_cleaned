# NCAA 2026 Stage 2 Prediction

This project generates predictions for the 2026 NCAA men's and women's tournament Stage 2 submission format.

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

```bash
pip install -r requirements.txt
```

Place the Kaggle NCAA CSV files in `data/raw/` before running the pipeline.
The raw data files are intentionally excluded from git because one source file
is larger than GitHub's normal 100 MB file limit.

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

## Notes

- The old exploratory `ncaa_*.py` scripts were removed to keep one maintainable pipeline.
- Historical submissions are stored under `outputs/`.
- Existing saved model files are kept locally under `models/legacy_saved_models/`, but the current pipeline retrains from source CSV data.
