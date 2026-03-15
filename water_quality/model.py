from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


@dataclass
class TrainResult:
    accuracy: float
    report: str
    feature_columns: List[str]
    label_column: str


class WaterQualityModel:
    def __init__(self) -> None:
        self.model: Optional[RandomForestClassifier] = None
        self.feature_columns: List[str] = []
        self.label_column: Optional[str] = None

    def train(
        self,
        data_path: Path,
        label_col: str = "quality_level",
        feature_cols: Optional[List[str]] = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> TrainResult:
        df = pd.read_csv(data_path)
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in {data_path}")

        if feature_cols:
            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                raise ValueError(f"Feature columns missing: {missing}")
            self.feature_columns = feature_cols
        else:
            self.feature_columns = [c for c in df.columns if c != label_col]

        self.label_column = label_col
        x = df[self.feature_columns]
        y = df[label_col]

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state, stratify=y
        )

        self.model = RandomForestClassifier(n_estimators=300, random_state=random_state)
        self.model.fit(x_train, y_train)

        y_pred = self.model.predict(x_test)
        acc = float(accuracy_score(y_test, y_pred))
        report = str(classification_report(y_test, y_pred))

        return TrainResult(
            accuracy=acc,
            report=report,
            feature_columns=self.feature_columns,
            label_column=label_col,
        )

    def predict(self, features: Dict[str, float]) -> Dict[str, object]:
        if self.model is None:
            raise RuntimeError("Model not loaded or trained.")
        missing = [c for c in self.feature_columns if c not in features]
        if missing:
            raise ValueError(f"Missing features: {missing}")

        row = pd.DataFrame([{c: float(features[c]) for c in self.feature_columns}])
        pred = self.model.predict(row)[0]
        result: Dict[str, object] = {"predicted_level": str(pred)}

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(row)[0]
            classes = [str(c) for c in self.model.classes_]
            result["probabilities"] = {cls: float(p) for cls, p in zip(classes, proba)}
        return result

    def save(self, model_path: Path) -> None:
        if self.model is None:
            raise RuntimeError("No model to save.")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "feature_columns": self.feature_columns,
            "label_column": self.label_column,
        }
        with model_path.open("wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, model_path: Path) -> "WaterQualityModel":
        with model_path.open("rb") as f:
            payload = pickle.load(f)

        obj = cls()
        obj.model = payload["model"]
        obj.feature_columns = payload["feature_columns"]
        obj.label_column = payload["label_column"]
        return obj
