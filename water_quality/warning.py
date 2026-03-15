"""水质预警模块。

基于实时数据和历史趋势，生成多级预警（蓝/黄/橙/红），
支持超标预警、趋势恶化预警、突变预警。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .anomaly import detect_exceedance, detect_sudden_change
from .fusion import compute_wqi
from .standards import LEVELS, STANDARDS
from .trend import _mann_kendall, _sen_slope


# ---------------------------------------------------------------------------
# 预警等级
# ---------------------------------------------------------------------------

ALERT_LEVELS = ["blue", "yellow", "orange", "red"]

ALERT_LABELS = {
    "blue": "蓝色预警（关注）",
    "yellow": "黄色预警（警告）",
    "orange": "橙色预警（严重）",
    "red": "红色预警（紧急）",
}


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class AlertItem:
    """单条预警。"""

    alert_level: str  # blue / yellow / orange / red
    indicator: str  # 触发指标（或 "overall"）
    alert_type: str  # "exceedance" / "trend" / "sudden_change" / "wqi"
    value: float
    threshold: float
    message: str
    timestamp: Optional[str] = None


@dataclass
class WarningReport:
    """预警报告。"""

    alerts: List[AlertItem] = field(default_factory=list)
    highest_level: str = "blue"
    summary: str = ""


# ---------------------------------------------------------------------------
# 实时超标预警
# ---------------------------------------------------------------------------


def _check_exceedance_alerts(
    values: Dict[str, float],
) -> List[AlertItem]:
    """基于当前值和国标阈值生成超标预警。"""
    alerts: List[AlertItem] = []

    for ind, val in values.items():
        if ind not in STANDARDS:
            continue

        rule = STANDARDS[ind]
        if "range" in rule:
            lo = float(rule["range"]["min"])  # type: ignore[index]
            hi = float(rule["range"]["max"])  # type: ignore[index]
            if val < lo or val > hi:
                alerts.append(AlertItem(
                    alert_level="red",
                    indicator=ind,
                    alert_type="exceedance",
                    value=val,
                    threshold=lo if val < lo else hi,
                    message=f"{ind} 值 {val} 超出标准范围 [{lo}, {hi}]",
                ))
            continue

        thresholds = rule.get("thresholds", {})
        direction = rule.get("direction", "max")

        # 逐级检查
        for level_idx, level in enumerate(LEVELS):
            th = thresholds.get(level)  # type: ignore[union-attr]
            if th is None:
                continue
            th = float(th)  # type: ignore[arg-type]

            exceeded = (val > th) if direction == "max" else (val < th)
            if not exceeded:
                break
        else:
            # 超出所有级别 → 红色预警
            level_idx = len(LEVELS)

        if level_idx >= 4:  # 超 V 类
            alerts.append(AlertItem(
                alert_level="red",
                indicator=ind,
                alert_type="exceedance",
                value=val,
                threshold=float(thresholds.get("V", 0)),  # type: ignore[union-attr, arg-type]
                message=f"{ind} 值 {val} 超出 V 类标准",
            ))
        elif level_idx == 3:  # IV-V 类
            alerts.append(AlertItem(
                alert_level="orange",
                indicator=ind,
                alert_type="exceedance",
                value=val,
                threshold=float(thresholds.get("IV", 0)),  # type: ignore[union-attr, arg-type]
                message=f"{ind} 值 {val} 达到 IV-V 类水质",
            ))
        elif level_idx == 2:  # III 类边缘
            alerts.append(AlertItem(
                alert_level="yellow",
                indicator=ind,
                alert_type="exceedance",
                value=val,
                threshold=float(thresholds.get("III", 0)),  # type: ignore[union-attr, arg-type]
                message=f"{ind} 值 {val} 接近 III 类标准临界值",
            ))

    return alerts


# ---------------------------------------------------------------------------
# 趋势恶化预警
# ---------------------------------------------------------------------------


def _check_trend_alerts(
    df: pd.DataFrame,
    indicators: Optional[List[str]] = None,
    significance: float = 0.05,
) -> List[AlertItem]:
    """基于近期历史数据的趋势恶化预警。"""
    if indicators is None:
        indicators = [c for c in ["ph", "do", "codmn", "nh3n", "tp"] if c in df.columns]

    alerts: List[AlertItem] = []
    direction_map = {"do": "up", "codmn": "down", "nh3n": "down", "tp": "down", "ph": "stable"}

    for ind in indicators:
        if ind not in df.columns:
            continue

        values = df[ind].dropna().to_numpy(dtype=float)
        if len(values) < 15:
            continue

        # 使用最近的数据进行趋势检测
        recent = values[-min(60, len(values)):]
        z, p = _mann_kendall(recent)
        slope = _sen_slope(recent)

        if p > significance:
            continue

        good_dir = direction_map.get(ind, "down")
        if good_dir == "up" and slope < 0:
            level = "orange" if abs(slope) > 0.05 else "yellow"
            alerts.append(AlertItem(
                alert_level=level,
                indicator=ind,
                alert_type="trend",
                value=slope,
                threshold=0.0,
                message=f"{ind} 呈显著下降趋势（斜率 {slope:.4f}），水质可能恶化",
            ))
        elif good_dir == "down" and slope > 0:
            level = "orange" if abs(slope) > 0.05 else "yellow"
            alerts.append(AlertItem(
                alert_level=level,
                indicator=ind,
                alert_type="trend",
                value=slope,
                threshold=0.0,
                message=f"{ind} 呈显著上升趋势（斜率 {slope:.4f}），水质可能恶化",
            ))

    return alerts


# ---------------------------------------------------------------------------
# 突变预警
# ---------------------------------------------------------------------------


def _check_sudden_change_alerts(
    df: pd.DataFrame,
    indicators: Optional[List[str]] = None,
    look_back: int = 5,
) -> List[AlertItem]:
    """检测最近 look_back 条数据是否存在突变。"""
    if indicators is None:
        indicators = [c for c in ["ph", "do", "codmn", "nh3n", "tp"] if c in df.columns]

    alerts: List[AlertItem] = []

    for ind in indicators:
        if ind not in df.columns or len(df) < 20:
            continue

        values = df[ind].dropna().to_numpy(dtype=float)
        if len(values) < 20:
            continue

        # 检查最近几个点
        baseline = values[:-look_back]
        recent = values[-look_back:]

        mean_base = float(np.mean(baseline))
        std_base = float(np.std(baseline))
        if std_base == 0:
            continue

        for v in recent:
            deviation = abs(v - mean_base) / std_base
            if deviation > 4.0:
                alerts.append(AlertItem(
                    alert_level="red",
                    indicator=ind,
                    alert_type="sudden_change",
                    value=float(v),
                    threshold=mean_base,
                    message=f"{ind} 突变：当前值 {v:.4f}，偏离基线均值 {mean_base:.4f} 达 {deviation:.1f} 倍标准差",
                ))
                break
            elif deviation > 3.0:
                alerts.append(AlertItem(
                    alert_level="orange",
                    indicator=ind,
                    alert_type="sudden_change",
                    value=float(v),
                    threshold=mean_base,
                    message=f"{ind} 突变预警：偏离基线 {deviation:.1f} 倍标准差",
                ))
                break

    return alerts


# ---------------------------------------------------------------------------
# WQI 综合预警
# ---------------------------------------------------------------------------


def _check_wqi_alert(values: Dict[str, float]) -> List[AlertItem]:
    """基于综合水质指数的预警。"""
    result = compute_wqi(values)
    alerts: List[AlertItem] = []

    if result.wqi < 25:
        alerts.append(AlertItem(
            alert_level="red",
            indicator="overall",
            alert_type="wqi",
            value=result.wqi,
            threshold=25.0,
            message=f"综合水质指数 WQI={result.wqi:.1f}（{result.level}类），水质极差",
        ))
    elif result.wqi < 50:
        alerts.append(AlertItem(
            alert_level="orange",
            indicator="overall",
            alert_type="wqi",
            value=result.wqi,
            threshold=50.0,
            message=f"综合水质指数 WQI={result.wqi:.1f}（{result.level}类），水质较差",
        ))
    elif result.wqi < 70:
        alerts.append(AlertItem(
            alert_level="yellow",
            indicator="overall",
            alert_type="wqi",
            value=result.wqi,
            threshold=70.0,
            message=f"综合水质指数 WQI={result.wqi:.1f}（{result.level}类），需关注",
        ))

    return alerts


# ---------------------------------------------------------------------------
# 综合预警入口
# ---------------------------------------------------------------------------


def generate_warnings(
    current_values: Dict[str, float],
    history_df: Optional[pd.DataFrame] = None,
    timestamp: Optional[str] = None,
) -> WarningReport:
    """生成综合预警报告。

    Parameters
    ----------
    current_values : dict
        当前时刻各指标值。
    history_df : DataFrame, optional
        历史数据（用于趋势和突变预警）。
    timestamp : str, optional
        当前时间戳。

    Returns
    -------
    WarningReport
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    all_alerts: List[AlertItem] = []

    # 1. 超标预警
    all_alerts.extend(_check_exceedance_alerts(current_values))

    # 2. WQI 预警
    all_alerts.extend(_check_wqi_alert(current_values))

    # 3. 趋势和突变预警（需要历史数据）
    if history_df is not None and len(history_df) >= 15:
        all_alerts.extend(_check_trend_alerts(history_df))
        all_alerts.extend(_check_sudden_change_alerts(history_df))

    # 设置时间戳
    for alert in all_alerts:
        alert.timestamp = timestamp

    # 确定最高预警级别
    highest = "blue"
    for alert in all_alerts:
        if ALERT_LEVELS.index(alert.alert_level) > ALERT_LEVELS.index(highest):
            highest = alert.alert_level

    # 生成摘要
    summary_lines: List[str] = []
    if not all_alerts:
        summary_lines.append("所有指标正常，无预警。")
    else:
        summary_lines.append(f"预警级别：{ALERT_LABELS[highest]}")
        summary_lines.append(f"共 {len(all_alerts)} 条预警信息：")
        for alert in all_alerts:
            summary_lines.append(f"  [{ALERT_LABELS[alert.alert_level]}] {alert.message}")

    return WarningReport(
        alerts=all_alerts,
        highest_level=highest,
        summary="\n".join(summary_lines),
    )
