from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .model import WaterQualityModel
from .standards import evaluate_level


DEFAULT_INDICATORS = ["ph", "do", "codmn", "nh3n", "tp"]
DEFAULT_CONTEXT = ["tide_stage", "tide_height", "temperature", "salinity"]


def load_measurements(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def predict_dataframe(
    df: pd.DataFrame,
    model_path: Path,
    indicators: Optional[List[str]] = None,
) -> pd.DataFrame:
    model = WaterQualityModel.load(model_path)
    features = model.feature_columns
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in input data: {missing}")

    x = df[features]
    preds = model.model.predict(x)  # type: ignore[union-attr]
    result = df.copy()
    result["predicted_level"] = preds

    if hasattr(model.model, "predict_proba"):  # type: ignore[union-attr]
        proba = model.model.predict_proba(x)  # type: ignore[union-attr]
        classes = [str(c) for c in model.model.classes_]  # type: ignore[union-attr]
        for idx, cls in enumerate(classes):
            result[f"prob_{cls}"] = proba[:, idx]

    use_indicators = indicators or DEFAULT_INDICATORS
    std_levels: List[str] = []
    for _, row in result.iterrows():
        values: Dict[str, float] = {}
        for k in use_indicators:
            if k in row:
                values[k] = float(row[k])
        std_levels.append(evaluate_level(values)["overall_level"])
    result["standard_level"] = std_levels
    return result


def summarize_by_tide(df: pd.DataFrame) -> pd.DataFrame:
    if "tide_stage" not in df.columns:
        return pd.DataFrame()
    grouped = df.groupby("tide_stage")
    summary = grouped.agg(
        count=("predicted_level", "count"),
        avg_do=("do", "mean"),
        avg_codmn=("codmn", "mean"),
        avg_nh3n=("nh3n", "mean"),
        avg_tp=("tp", "mean"),
    )
    return summary.reset_index()
