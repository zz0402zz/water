"""历史水质变化规律和趋势性分析模块。

提供时间序列趋势检测（Mann-Kendall）、季节性分解、移动平均、
变化率计算等功能，用于分析水质指标的长期演变规律。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class TrendResult:
    """单个指标的趋势分析结果。"""

    indicator: str
    trend: str  # "improving", "deteriorating", "stable"
    slope: float  # 每单位时间的变化速率
    p_value: float  # Mann-Kendall 检验 p 值
    mean: float
    std: float
    min_val: float
    max_val: float
    change_rate: float  # 整体变化率 (%)


@dataclass
class SeasonalResult:
    """季节性分析结果。"""

    indicator: str
    seasonal_strength: float  # 季节性强度 (0-1)
    peak_month: int  # 高峰月份（对于 max 指标：最差月；min 指标：最差月）
    trough_month: int  # 低谷月份


@dataclass
class TrendReport:
    """完整趋势分析报告。"""

    trends: Dict[str, TrendResult] = field(default_factory=dict)
    seasonals: Dict[str, SeasonalResult] = field(default_factory=dict)
    summary: str = ""


# ---------------------------------------------------------------------------
# Mann-Kendall 趋势检验（简化版）
# ---------------------------------------------------------------------------


def _mann_kendall(series: np.ndarray) -> Tuple[float, float]:
    """简化版 Mann-Kendall 趋势检验。

    返回 (S 统计量的标准化 Z, 近似 p 值)。
    """
    n = len(series)
    if n < 10:
        return 0.0, 1.0

    s = 0
    for k in range(n - 1):
        for j in range(k + 1, n):
            diff = series[j] - series[k]
            if diff > 0:
                s += 1
            elif diff < 0:
                s -= 1

    var_s = n * (n - 1) * (2 * n + 5) / 18.0
    if var_s == 0:
        return 0.0, 1.0

    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0

    # 近似 p 值（使用正态分布）
    p_value = float(2.0 * (1.0 - _norm_cdf(abs(z))))
    return float(z), p_value


def _norm_cdf(x: float) -> float:
    """标准正态分布 CDF 近似。"""
    return float(0.5 * (1.0 + _erf(x / np.sqrt(2.0))))


def _erf(x: float) -> float:
    """误差函数近似（Abramowitz and Stegun）。"""
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return float(sign * y)


def _sen_slope(series: np.ndarray) -> float:
    """Sen's slope 估计量，用于估计趋势的斜率。"""
    n = len(series)
    if n < 2:
        return 0.0
    slopes: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            if j != i:
                slopes.append((series[j] - series[i]) / (j - i))
    return float(np.median(slopes)) if slopes else 0.0


# ---------------------------------------------------------------------------
# 移动平均
# ---------------------------------------------------------------------------


def moving_average(series: pd.Series, window: int = 7) -> pd.Series:
    """计算移动平均。"""
    return series.rolling(window=window, min_periods=1).mean()


# ---------------------------------------------------------------------------
# 主分析函数
# ---------------------------------------------------------------------------

INDICATORS = ["ph", "do", "codmn", "nh3n", "tp"]

# do 是 "越高越好"，其他指标是 "越低越好" 或范围类
_IMPROVING_DIRECTION = {
    "ph": "stable",  # pH 稳定在 6-9 即可
    "do": "up",  # 溶解氧越高水质越好
    "codmn": "down",  # 越低越好
    "nh3n": "down",
    "tp": "down",
}


def analyze_trend(
    df: pd.DataFrame,
    time_col: str = "date",
    indicators: Optional[List[str]] = None,
    significance: float = 0.05,
) -> TrendReport:
    """对历史水质数据执行趋势分析。

    Parameters
    ----------
    df : DataFrame
        必须包含时间列和指标列。
    time_col : str
        时间列名称。
    indicators : list, optional
        要分析的指标列表，默认分析所有已知指标。
    significance : float
        Mann-Kendall 检验显著性水平。

    Returns
    -------
    TrendReport
    """
    if indicators is None:
        indicators = [c for c in INDICATORS if c in df.columns]

    df = df.copy().sort_values(time_col).reset_index(drop=True)
    report = TrendReport()

    for ind in indicators:
        if ind not in df.columns:
            continue

        values = df[ind].dropna().to_numpy(dtype=float)
        if len(values) < 5:
            continue

        z, p = _mann_kendall(values)
        slope = _sen_slope(values)
        mean_val = float(np.mean(values))
        std_val = float(np.std(values))
        min_val = float(np.min(values))
        max_val = float(np.max(values))

        # 变化率：首尾移动平均对比
        head_avg = float(np.mean(values[:max(1, len(values) // 10)]))
        tail_avg = float(np.mean(values[-max(1, len(values) // 10):]))
        change_rate = ((tail_avg - head_avg) / head_avg * 100) if head_avg != 0 else 0.0

        # 判断趋势方向
        direction = _IMPROVING_DIRECTION.get(ind, "down")
        if p > significance:
            trend_label = "stable"
        elif direction == "up":
            trend_label = "improving" if slope > 0 else "deteriorating"
        elif direction == "down":
            trend_label = "improving" if slope < 0 else "deteriorating"
        else:  # stable (pH)
            trend_label = "stable" if abs(slope) < 0.01 else ("shifting" if slope > 0 else "shifting")

        report.trends[ind] = TrendResult(
            indicator=ind,
            trend=trend_label,
            slope=slope,
            p_value=p,
            mean=mean_val,
            std=std_val,
            min_val=min_val,
            max_val=max_val,
            change_rate=change_rate,
        )

    # 季节性分析（如果有时间列且数据量够）
    if time_col in df.columns:
        try:
            dates = pd.to_datetime(df[time_col], errors="coerce")
            if len(dates) >= 24:  # 至少两年数据
                for ind in indicators:
                    if ind not in df.columns:
                        continue
                    seasonal = _analyze_seasonal(dates, df[ind].to_numpy(dtype=float))
                    if seasonal:
                        report.seasonals[ind] = SeasonalResult(
                            indicator=ind,
                            seasonal_strength=seasonal[0],
                            peak_month=seasonal[1],
                            trough_month=seasonal[2],
                        )
        except (ValueError, TypeError):
            pass

    # 生成文字摘要
    report.summary = _build_summary(report)
    return report


def _analyze_seasonal(
    dates: pd.Series, values: np.ndarray
) -> Optional[Tuple[float, int, int]]:
    """简单的月度季节性分析。"""
    try:
        values = np.asarray(values, dtype=float)
        monthly_means: Dict[int, List[float]] = {}
        for d, v in zip(dates, values):
            if pd.isna(d) or np.isnan(v):
                continue
            month = int(pd.Timestamp(d).month)
            monthly_means.setdefault(month, []).append(float(v))

        if len(monthly_means) < 6:
            return None

        month_avg = {m: float(np.mean(vals)) for m, vals in monthly_means.items()}
        overall_mean = float(np.mean(list(month_avg.values())))
        if overall_mean == 0:
            return None

        # 季节性强度：月均值标准差 / 整体均值
        seasonal_std = float(np.std(list(month_avg.values())))
        strength = float(min(1.0, seasonal_std / abs(overall_mean)))

        peak_month = int(max(month_avg.items(), key=lambda kv: kv[1])[0])
        trough_month = int(min(month_avg.items(), key=lambda kv: kv[1])[0])

        return (strength, peak_month, trough_month)
    except Exception:
        return None


def _build_summary(report: TrendReport) -> str:
    """生成趋势分析文字摘要。"""
    lines: List[str] = []
    improving = [t.indicator for t in report.trends.values() if t.trend == "improving"]
    deteriorating = [t.indicator for t in report.trends.values() if t.trend == "deteriorating"]
    stable = [t.indicator for t in report.trends.values() if t.trend == "stable"]

    if improving:
        lines.append(f"改善趋势指标: {', '.join(improving)}")
    if deteriorating:
        lines.append(f"恶化趋势指标: {', '.join(deteriorating)}")
    if stable:
        lines.append(f"稳定指标: {', '.join(stable)}")

    for ind, s in report.seasonals.items():
        if s.seasonal_strength > 0.2:
            lines.append(f"{ind} 存在明显季节性变化（强度 {s.seasonal_strength:.2f}），"
                         f"高峰月: {s.peak_month}月, 低谷月: {s.trough_month}月")

    return "\n".join(lines) if lines else "数据不足，无法得出趋势结论。"
