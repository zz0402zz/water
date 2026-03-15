"""汛期/非汛期水质分析模块。

将监测数据按汛期（默认 5-10 月）和非汛期分组，比较各指标均值、
超标率、水质等级分布差异，输出汛期影响评估报告。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .standards import LEVELS, STANDARDS, _level_for_indicator


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

# 默认汛期月份（南方常见），可按流域自定义
DEFAULT_FLOOD_MONTHS = [5, 6, 7, 8, 9, 10]

INDICATORS = ["ph", "do", "codmn", "nh3n", "tp"]


@dataclass
class SeasonStats:
    """某一时段（汛期或非汛期）单个指标的统计信息。"""

    indicator: str
    count: int = 0
    mean: float = 0.0
    std: float = 0.0
    median: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    exceedance_rate: float = 0.0  # 超标率（相对 III 类）
    level_distribution: Dict[str, int] = field(default_factory=dict)


@dataclass
class SeasonComparison:
    """单个指标的汛期/非汛期对比结果。"""

    indicator: str
    flood_stats: SeasonStats
    dry_stats: SeasonStats
    mean_diff: float = 0.0         # 均值差异 (flood - dry)
    mean_diff_pct: float = 0.0     # 均值差异百分比
    exceedance_diff: float = 0.0   # 超标率差异
    significant: bool = False       # Mann-Whitney U 检验是否显著
    p_value: float = 1.0


@dataclass
class SeasonReport:
    """汛期/非汛期综合分析报告。"""

    flood_months: List[int] = field(default_factory=list)
    flood_count: int = 0
    dry_count: int = 0
    comparisons: Dict[str, SeasonComparison] = field(default_factory=dict)
    summary: str = ""


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _compute_stats(
    series: pd.Series,
    indicator: str,
    level: str = "III",
) -> SeasonStats:
    """计算单个指标在某一时段内的统计信息。"""
    values = series.dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return SeasonStats(indicator=indicator)

    # 水质等级分布
    level_dist: Dict[str, int] = {lv: 0 for lv in LEVELS}
    exceed_count = 0
    for v in values:
        lv = _level_for_indicator(indicator, float(v))
        level_dist[lv] = level_dist.get(lv, 0) + 1
        if LEVELS.index(lv) > LEVELS.index(level):
            exceed_count += 1

    return SeasonStats(
        indicator=indicator,
        count=len(values),
        mean=float(np.mean(values)),
        std=float(np.std(values)),
        median=float(np.median(values)),
        min_val=float(np.min(values)),
        max_val=float(np.max(values)),
        exceedance_rate=exceed_count / len(values) if len(values) > 0 else 0.0,
        level_distribution=level_dist,
    )


def _mann_whitney_u(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """简化版 Mann-Whitney U 检验（大样本正态近似）。

    返回 (U, p_value)。
    """
    n1, n2 = len(a), len(b)
    if n1 < 5 or n2 < 5:
        return 0.0, 1.0

    combined = np.concatenate([a, b])
    ranks = np.empty_like(combined, dtype=float)
    order = combined.argsort()
    ranks[order] = np.arange(1, len(combined) + 1, dtype=float)

    r1 = float(ranks[:n1].sum())
    u1 = r1 - n1 * (n1 + 1) / 2

    mu = n1 * n2 / 2.0
    sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    if sigma == 0:
        return u1, 1.0

    z = abs((u1 - mu) / sigma)
    # 正态近似 p 值
    p = float(2.0 * (1.0 - 0.5 * (1.0 + _erf(z / np.sqrt(2.0)))))
    return u1, max(0.0, min(1.0, p))


def _erf(x: float) -> float:
    """误差函数近似。"""
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    x = abs(float(x))
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return float(sign * y)


# ---------------------------------------------------------------------------
# 主分析函数
# ---------------------------------------------------------------------------


def analyze_season(
    df: pd.DataFrame,
    time_col: str = "date",
    indicators: Optional[List[str]] = None,
    flood_months: Optional[List[int]] = None,
    exceedance_level: str = "III",
    significance: float = 0.05,
) -> SeasonReport:
    """汛期/非汛期水质对比分析。

    Parameters
    ----------
    df : DataFrame
        水质监测数据，需包含时间列和指标列。
    time_col : str
        时间列名。
    indicators : list, optional
        要分析的指标，默认全部。
    flood_months : list, optional
        汛期月份列表，默认 [5,6,7,8,9,10]。
    exceedance_level : str
        超标对照等级。
    significance : float
        显著性水平。

    Returns
    -------
    SeasonReport
    """
    if flood_months is None:
        flood_months = DEFAULT_FLOOD_MONTHS
    if indicators is None:
        indicators = [c for c in INDICATORS if c in df.columns]

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])
    df["_month"] = df[time_col].dt.month

    flood_mask = df["_month"].isin(flood_months)
    df_flood = df[flood_mask]
    df_dry = df[~flood_mask]

    report = SeasonReport(
        flood_months=flood_months,
        flood_count=len(df_flood),
        dry_count=len(df_dry),
    )

    for ind in indicators:
        if ind not in df.columns:
            continue

        flood_stats = _compute_stats(df_flood[ind], ind, level=exceedance_level)
        dry_stats = _compute_stats(df_dry[ind], ind, level=exceedance_level)

        # 均值差异
        mean_diff = flood_stats.mean - dry_stats.mean
        base = dry_stats.mean if dry_stats.mean != 0 else 1.0
        mean_diff_pct = (mean_diff / abs(base)) * 100

        # Mann-Whitney U 检验
        flood_vals = df_flood[ind].dropna().to_numpy(dtype=float)
        dry_vals = df_dry[ind].dropna().to_numpy(dtype=float)
        _, p_val = _mann_whitney_u(flood_vals, dry_vals)

        report.comparisons[ind] = SeasonComparison(
            indicator=ind,
            flood_stats=flood_stats,
            dry_stats=dry_stats,
            mean_diff=mean_diff,
            mean_diff_pct=mean_diff_pct,
            exceedance_diff=flood_stats.exceedance_rate - dry_stats.exceedance_rate,
            significant=p_val < significance,
            p_value=p_val,
        )

    report.summary = _build_summary(report)
    return report


def _build_summary(report: SeasonReport) -> str:
    """生成汛期分析文字摘要。"""
    lines: List[str] = []
    lines.append(f"汛期月份: {report.flood_months}")
    lines.append(f"汛期样本数: {report.flood_count}, 非汛期样本数: {report.dry_count}")
    lines.append("")

    sig_worse: List[str] = []
    sig_better: List[str] = []

    for ind, comp in report.comparisons.items():
        if not comp.significant:
            continue
        # 对于 do，汛期降低=恶化；对于其他污染物，汛期升高=恶化
        if ind == "do":
            if comp.mean_diff < 0:
                sig_worse.append(f"{ind}(↓{abs(comp.mean_diff_pct):.1f}%)")
            else:
                sig_better.append(f"{ind}(↑{comp.mean_diff_pct:.1f}%)")
        else:
            if comp.mean_diff > 0:
                sig_worse.append(f"{ind}(↑{comp.mean_diff_pct:.1f}%)")
            else:
                sig_better.append(f"{ind}(↓{abs(comp.mean_diff_pct):.1f}%)")

    if sig_worse:
        lines.append(f"汛期显著恶化指标: {', '.join(sig_worse)}")
    if sig_better:
        lines.append(f"汛期显著改善指标: {', '.join(sig_better)}")
    if not sig_worse and not sig_better:
        lines.append("汛期与非汛期水质无显著性差异。")

    # 超标率对比
    exceed_lines: List[str] = []
    for ind, comp in report.comparisons.items():
        if comp.exceedance_diff > 0.05:  # 汛期超标率高出5%以上
            exceed_lines.append(
                f"  {ind}: 汛期超标率 {comp.flood_stats.exceedance_rate:.1%} "
                f"vs 非汛期 {comp.dry_stats.exceedance_rate:.1%}"
            )
    if exceed_lines:
        lines.append("\n汛期超标率明显偏高的指标:")
        lines.extend(exceed_lines)

    return "\n".join(lines)
