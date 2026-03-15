"""水质异常识别模块。

提供多种异常检测方法：Z-score、IQR、Isolation Forest，
以及基于水质标准的超标检测。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .standards import LEVELS, STANDARDS


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class AnomalyRecord:
    """单条异常记录。"""

    index: int  # 数据行索引
    indicator: str  # 指标名
    value: float  # 实际值
    method: str  # 检测方法
    severity: str  # "warning", "critical"
    reason: str  # 异常原因描述


@dataclass
class AnomalyReport:
    """异常检测报告。"""

    total_records: int = 0
    anomaly_count: int = 0
    anomalies: List[AnomalyRecord] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)  # 按指标统计异常数


# ---------------------------------------------------------------------------
# 基于统计的异常检测
# ---------------------------------------------------------------------------


def detect_zscore(
    series: pd.Series,
    indicator: str,
    threshold: float = 3.0,
) -> List[AnomalyRecord]:
    """Z-score 异常检测。

    |z| > threshold 视为异常。
    """
    values = series.dropna().to_numpy(dtype=float)
    if len(values) < 10:
        return []

    mean = float(np.mean(values))
    std = float(np.std(values))
    if std == 0:
        return []

    anomalies: List[AnomalyRecord] = []
    for idx, val in enumerate(values):
        z = abs((float(val) - mean) / std)
        if z > threshold:
            severity = "critical" if z > threshold * 1.5 else "warning"
            anomalies.append(AnomalyRecord(
                index=idx,
                indicator=indicator,
                value=float(val),
                method="z-score",
                severity=severity,
                reason=f"Z-score={z:.2f}（阈值 {threshold}），均值={mean:.4f}，标准差={std:.4f}",
            ))
    return anomalies


def detect_iqr(
    series: pd.Series,
    indicator: str,
    multiplier: float = 1.5,
) -> List[AnomalyRecord]:
    """IQR 四分位距异常检测。"""
    values = series.dropna().to_numpy(dtype=float)
    if len(values) < 10:
        return []

    q1 = float(np.quantile(values, 0.25))
    q3 = float(np.quantile(values, 0.75))
    iqr = q3 - q1
    if iqr == 0:
        return []

    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr

    anomalies: List[AnomalyRecord] = []
    for idx, val in enumerate(values):
        v = float(val)
        if v < lower or v > upper:
            severity = "critical" if (v < q1 - 3 * iqr or v > q3 + 3 * iqr) else "warning"
            anomalies.append(AnomalyRecord(
                index=idx,
                indicator=indicator,
                value=v,
                method="iqr",
                severity=severity,
                reason=f"超出 IQR 范围 [{lower:.4f}, {upper:.4f}]",
            ))
    return anomalies


# ---------------------------------------------------------------------------
# 基于标准阈值的超标检测
# ---------------------------------------------------------------------------


def detect_exceedance(
    series: pd.Series,
    indicator: str,
    level: str = "III",
) -> List[AnomalyRecord]:
    """基于国标阈值的超标检测。

    Parameters
    ----------
    level : str
        对照的水质类别（默认 III 类标准）。
    """
    if indicator not in STANDARDS:
        return []

    rule = STANDARDS[indicator]
    anomalies: List[AnomalyRecord] = []

    if "range" in rule:
        lo = float(rule["range"]["min"])  # type: ignore[index]
        hi = float(rule["range"]["max"])  # type: ignore[index]
        for idx, val in enumerate(series.dropna().to_numpy(dtype=float)):
            v = float(val)
            if v < lo or v > hi:
                anomalies.append(AnomalyRecord(
                    index=idx,
                    indicator=indicator,
                    value=v,
                    method="exceedance",
                    severity="critical",
                    reason=f"超出 {indicator} 标准范围 [{lo}, {hi}]",
                ))
        return anomalies

    thresholds = rule.get("thresholds", {})
    direction = rule.get("direction", "max")
    threshold_val = thresholds.get(level)  # type: ignore[union-attr]
    if threshold_val is None:
        return []
    threshold_val = float(threshold_val)  # type: ignore[arg-type]

    for idx, val in enumerate(series.dropna().to_numpy(dtype=float)):
        v = float(val)
        exceeded = (v > threshold_val) if direction == "max" else (v < threshold_val)
        if exceeded:
            # 判定严重程度
            level_idx = LEVELS.index(level)
            if level_idx < len(LEVELS) - 1:
                next_threshold = thresholds.get(LEVELS[level_idx + 1])  # type: ignore[union-attr]
                if next_threshold is not None:
                    next_exceeded = (v > float(next_threshold)) if direction == "max" else (v < float(next_threshold))
                    severity = "critical" if next_exceeded else "warning"
                else:
                    severity = "warning"
            else:
                severity = "critical"

            anomalies.append(AnomalyRecord(
                index=idx,
                indicator=indicator,
                value=v,
                method="exceedance",
                severity=severity,
                reason=f"超出{level}类标准（阈值 {threshold_val}，方向 {direction}）",
            ))
    return anomalies


# ---------------------------------------------------------------------------
# 突变检测（基于滑动窗口的均值突变）
# ---------------------------------------------------------------------------


def detect_sudden_change(
    series: pd.Series,
    indicator: str,
    window: int = 10,
    threshold_factor: float = 3.0,
) -> List[AnomalyRecord]:
    """滑动窗口突变检测。

    比较当前值与前 window 个值的均值，偏差超过 threshold_factor 倍标准差视为突变。
    """
    values = series.dropna().to_numpy(dtype=float)
    if len(values) < window + 5:
        return []

    anomalies: List[AnomalyRecord] = []
    for i in range(window, len(values)):
        win = values[i - window: i]
        win_mean = float(np.mean(win))
        win_std = float(np.std(win))
        if win_std == 0:
            continue

        deviation = abs(values[i] - win_mean) / win_std
        if deviation > threshold_factor:
            severity = "critical" if deviation > threshold_factor * 2 else "warning"
            anomalies.append(AnomalyRecord(
                index=i,
                indicator=indicator,
                value=float(values[i]),
                method="sudden_change",
                severity=severity,
                reason=f"突变检测：偏差 {deviation:.2f} 倍标准差（窗口均值 {win_mean:.4f}）",
            ))
    return anomalies


# ---------------------------------------------------------------------------
# 综合异常检测
# ---------------------------------------------------------------------------

INDICATORS = ["ph", "do", "codmn", "nh3n", "tp"]


def detect_anomalies(
    df: pd.DataFrame,
    indicators: Optional[List[str]] = None,
    methods: Optional[List[str]] = None,
    exceedance_level: str = "III",
) -> AnomalyReport:
    """综合异常检测。

    Parameters
    ----------
    df : DataFrame
        水质数据。
    indicators : list, optional
        要检测的指标，默认所有。
    methods : list, optional
        检测方法列表：["zscore", "iqr", "exceedance", "sudden_change"]，默认全部。
    exceedance_level : str
        超标检测对照的水质类别。

    Returns
    -------
    AnomalyReport
    """
    if indicators is None:
        indicators = [c for c in INDICATORS if c in df.columns]
    if methods is None:
        methods = ["zscore", "iqr", "exceedance", "sudden_change"]

    report = AnomalyReport(total_records=len(df))
    all_anomalies: List[AnomalyRecord] = []

    for ind in indicators:
        if ind not in df.columns:
            continue

        series = df[ind]
        ind_anomalies: List[AnomalyRecord] = []

        if "zscore" in methods:
            ind_anomalies.extend(detect_zscore(series, ind))
        if "iqr" in methods:
            ind_anomalies.extend(detect_iqr(series, ind))
        if "exceedance" in methods:
            ind_anomalies.extend(detect_exceedance(series, ind, level=exceedance_level))
        if "sudden_change" in methods:
            ind_anomalies.extend(detect_sudden_change(series, ind))

        # 去重（同一行同一指标保留最严重的）
        deduped = _deduplicate(ind_anomalies)
        report.summary[ind] = len(deduped)
        all_anomalies.extend(deduped)

    report.anomalies = sorted(all_anomalies, key=lambda a: (a.index, a.indicator))
    report.anomaly_count = len(report.anomalies)
    return report


def _deduplicate(records: List[AnomalyRecord]) -> List[AnomalyRecord]:
    """同一行索引保留最严重的异常记录。"""
    by_index: Dict[int, AnomalyRecord] = {}
    severity_rank = {"critical": 2, "warning": 1}

    for r in records:
        existing = by_index.get(r.index)
        if existing is None or severity_rank.get(r.severity, 0) > severity_rank.get(existing.severity, 0):
            by_index[r.index] = r
    return list(by_index.values())
