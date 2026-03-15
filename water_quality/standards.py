from __future__ import annotations

from typing import Dict, Optional


LEVELS = ["I", "II", "III", "IV", "V"]

# Reference style: surface water quality classification (common indicators).
STANDARDS: Dict[str, Dict[str, object]] = {
    "ph": {
        "name": "pH",
        "unit": "-",
        "range": {"min": 6.0, "max": 9.0},
        "note": "All classes use the same pH range.",
    },
    "do": {
        "name": "Dissolved Oxygen",
        "unit": "mg/L",
        "thresholds": {"I": 7.5, "II": 6.0, "III": 5.0, "IV": 3.0, "V": 2.0},
        "direction": "min",
    },
    "codmn": {
        "name": "Permanganate Index (CODMn)",
        "unit": "mg/L",
        "thresholds": {"I": 2.0, "II": 4.0, "III": 6.0, "IV": 10.0, "V": 15.0},
        "direction": "max",
    },
    "nh3n": {
        "name": "Ammonia Nitrogen (NH3-N)",
        "unit": "mg/L",
        "thresholds": {"I": 0.15, "II": 0.5, "III": 1.0, "IV": 1.5, "V": 2.0},
        "direction": "max",
    },
    "tp": {
        "name": "Total Phosphorus (TP)",
        "unit": "mg/L",
        "thresholds": {"I": 0.02, "II": 0.1, "III": 0.2, "IV": 0.3, "V": 0.4},
        "direction": "max",
    },
}


def get_standards() -> Dict[str, Dict[str, object]]:
    return STANDARDS


def explain_indicator(indicator: str) -> Dict[str, object]:
    key = indicator.lower()
    if key not in STANDARDS:
        raise KeyError(f"Unknown indicator: {indicator}")
    return STANDARDS[key]


def _level_for_indicator(indicator: str, value: float) -> str:
    rule = explain_indicator(indicator)

    if "range" in rule:
        lower = float(rule["range"]["min"])  # type: ignore[index]
        upper = float(rule["range"]["max"])  # type: ignore[index]
        return "III" if lower <= value <= upper else "V"

    thresholds = rule["thresholds"]  # type: ignore[index]
    direction = rule["direction"]  # type: ignore[index]

    if direction == "max":
        for level in LEVELS:
            if value <= float(thresholds[level]):  # type: ignore[index]
                return level
        return "V"

    for level in LEVELS:
        if value >= float(thresholds[level]):  # type: ignore[index]
            return level
    return "V"


def evaluate_level(values: Dict[str, float]) -> Dict[str, object]:
    details: Dict[str, str] = {}
    worst_idx = 0

    for indicator, value in values.items():
        if indicator not in STANDARDS:
            continue
        level = _level_for_indicator(indicator, float(value))
        details[indicator] = level
        worst_idx = max(worst_idx, LEVELS.index(level))

    overall = LEVELS[worst_idx] if details else "UNKNOWN"
    return {"overall_level": overall, "by_indicator": details}
