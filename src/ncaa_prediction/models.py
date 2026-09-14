from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS


@dataclass
class EnsembleModel:
    logistic: CalibratedClassifierCV
    gradient_boosting: GradientBoostingClassifier
    clip_min: float = 0.001
    clip_max: float = 0.999

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        features = frame[FEATURE_COLUMNS].fillna(0)
        lr_prob = self.logistic.predict_proba(features)[:, 1]
        gb_prob = self.gradient_boosting.predict_proba(features)[:, 1]
        return np.clip((lr_prob + gb_prob) / 2, self.clip_min, self.clip_max)


def train_ensemble(train_df: pd.DataFrame, clip_min: float = 0.001,
                   clip_max: float = 0.999, random_state: int = 42) -> EnsembleModel:
    if train_df.empty or "Result" not in train_df:
        raise ValueError("No tournament training games found for the selected seasons")
    counts = train_df["Result"].value_counts()
    if set(counts.index) != {0, 1} or counts.min() < 5:
        raise ValueError("Five-fold calibration requires at least five games for each binary outcome")
    x_train = train_df[FEATURE_COLUMNS].fillna(0)
    y_train = train_df["Result"]

    logistic = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=0.1,
                solver="liblinear",
                max_iter=2000,
                random_state=random_state,
            )),
        ]
    )
    logistic_calibrated = CalibratedClassifierCV(logistic, method="isotonic", cv=5)
    logistic_calibrated.fit(x_train, y_train)

    gradient_boosting = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        min_samples_split=20,
        min_samples_leaf=8,
        max_features="sqrt",
        random_state=random_state,
    )
    gradient_boosting.fit(x_train, y_train)

    return EnsembleModel(logistic_calibrated, gradient_boosting, clip_min, clip_max)


def evaluate_model(model: EnsembleModel, val_df: pd.DataFrame) -> Dict[str, float]:
    if val_df.empty:
        raise ValueError("No tournament validation games found for the selected seasons")
    y_val = val_df["Result"]
    pred = model.predict_proba(val_df)
    labels = (pred >= 0.5).astype(int)
    return {
        "log_loss": float(log_loss(y_val, pred, labels=[0, 1])),
        "brier": float(brier_score_loss(y_val, pred)),
        "accuracy": float(accuracy_score(y_val, labels)),
    }
