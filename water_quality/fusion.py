"""多维度数据融合模块。

将多个水质指标融合为综合水质指数（WQI），支持加权融合、
主成分分析（PCA）降维、以及多维度综合评价。
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

DEFAULT_WEIGHTS: Dict[str, float] = {
    "ph": 0.10,
    "do": 0.25,
    "codmn": 0.20,
    "nh3n": 0.25,
    "tp": 0.20,
}


@dataclass
class FusionResult:
    """单条数据的融合评价结果。"""

    wqi: float  # 综合水质指数 (0-100, 越高越好)
    level: str  # 综合等级
    indicator_scores: Dict[str, float] = field(default_factory=dict)  # 各指标归一化得分


@dataclass
class FusionReport:
    """批量数据的融合分析报告。"""

    results: List[FusionResult] = field(default_factory=list)
    wqi_mean: float = 0.0
    wqi_std: float = 0.0
    level_distribution: Dict[str, int] = field(default_factory=dict)
    correlation_matrix: Optional[pd.DataFrame] = None
    pca_explained_variance: Optional[List[float]] = None


# ---------------------------------------------------------------------------
# 指标归一化（映射到 0-100 分）
# ---------------------------------------------------------------------------


def _normalize_indicator(indicator: str, value: float) -> float:
    """将单个指标值归一化到 0-100 分（越高越好）。"""
    if indicator not in STANDARDS:
        return 50.0

    rule = STANDARDS[indicator]

    if "range" in rule:
        lo = float(rule["range"]["min"])  # type: ignore[index]
        hi = float(rule["range"]["max"])  # type: ignore[index]
        mid = (lo + hi) / 2
        deviation = abs(value - mid) / ((hi - lo) / 2) if hi != lo else 0.0
        return max(0.0, min(100.0, 100.0 * (1.0 - deviation)))

    thresholds = rule["thresholds"]  # type: ignore[index]
    direction = rule["direction"]

    # 获取 I 类和 V 类阈值
    best = float(thresholds["I"])  # type: ignore[index]
    worst = float(thresholds["V"])  # type: ignore[index]

    if direction == "max":
        # 越小越好：best 是最小的阈值，worst 是最大的
        if value <= best:
            return 100.0
        if value >= worst:
            return 0.0
        return max(0.0, 100.0 * (1.0 - (value - best) / (worst - best)))
    else:
        # 越大越好（如溶解氧）：best 是最大的阈值
        if value >= best:
            return 100.0
        if value <= worst:
            return 0.0
        return max(0.0, 100.0 * (value - worst) / (best - worst))


def _score_to_level(wqi: float) -> str:
    """WQI 分数转换为水质等级。"""
    if wqi >= 90:
        return "I"
    if wqi >= 70:
        return "II"
    if wqi >= 50:
        return "III"
    if wqi >= 25:
        return "IV"
    return "V"


# ---------------------------------------------------------------------------
# 加权综合评价
# ---------------------------------------------------------------------------


def compute_wqi(
    values: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> FusionResult:
    """计算加权水质指数（WQI）。

    Parameters
    ----------
    values : dict
        各指标的测量值，如 {"ph": 7.0, "do": 8.0, ...}。
    weights : dict, optional
        各指标权重，默认使用 DEFAULT_WEIGHTS。

    Returns
    -------
    FusionResult
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    indicator_scores: Dict[str, float] = {}
    total_weight = 0.0
    weighted_sum = 0.0

    for ind, val in values.items():
        if ind not in weights:
            continue
        score = _normalize_indicator(ind, val)
        indicator_scores[ind] = score
        w = weights[ind]
        weighted_sum += score * w
        total_weight += w

    wqi = weighted_sum / total_weight if total_weight > 0 else 0.0
    return FusionResult(
        wqi=round(wqi, 2),
        level=_score_to_level(wqi),
        indicator_scores=indicator_scores,
    )


# ---------------------------------------------------------------------------
# 批量融合分析
# ---------------------------------------------------------------------------


def batch_fusion(
    df: pd.DataFrame,
    indicators: Optional[List[str]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> FusionReport:
    """对批量数据进行融合分析。

    Parameters
    ----------
    df : DataFrame
        水质数据。
    indicators : list, optional
        指标列表。
    weights : dict, optional
        权重。

    Returns
    -------
    FusionReport
    """
    if indicators is None:
        indicators = [c for c in DEFAULT_WEIGHTS if c in df.columns]
    if weights is None:
        weights = {k: v for k, v in DEFAULT_WEIGHTS.items() if k in indicators}

    report = FusionReport()

    # 逐行计算 WQI
    for _, row in df.iterrows():
        vals = {ind: float(row[ind]) for ind in indicators if ind in row.index and pd.notna(row[ind])}
        if not vals:
            continue
        result = compute_wqi(vals, weights)
        report.results.append(result)

    if report.results:
        wqis = [r.wqi for r in report.results]
        report.wqi_mean = round(float(np.mean(wqis)), 2)
        report.wqi_std = round(float(np.std(wqis)), 2)

        for r in report.results:
            report.level_distribution[r.level] = report.level_distribution.get(r.level, 0) + 1

    # 相关性矩阵
    available = [c for c in indicators if c in df.columns]
    if len(available) >= 2:
        report.correlation_matrix = df[available].corr()

    # PCA 分析
    if len(available) >= 2 and len(df) >= 10:
        report.pca_explained_variance = _pca_analysis(df[available].dropna())

    return report


def _pca_analysis(df: pd.DataFrame) -> Optional[List[float]]:
    """简单的 PCA 分析，返回各主成分的方差解释比例。"""
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaled = scaler.fit_transform(df)

        n_components = min(len(df.columns), len(df))
        pca = PCA(n_components=n_components)
        pca.fit(scaled)

        return [round(float(v), 4) for v in pca.explained_variance_ratio_]
    except Exception:
        return None
