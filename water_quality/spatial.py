"""水质空间关联分析模块。

分析多站点间水质指标的空间相关性、上下游传递关系、
聚类分组以及空间异质性，用于识别污染源和扩散规律。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .standards import _level_for_indicator


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

INDICATORS = ["ph", "do", "codmn", "nh3n", "tp"]


@dataclass
class StationProfile:
    """单站点统计概况。"""

    station: str
    record_count: int = 0
    indicator_means: Dict[str, float] = field(default_factory=dict)
    indicator_stds: Dict[str, float] = field(default_factory=dict)
    dominant_level: str = "III"  # 出现最多的水质等级


@dataclass
class StationPair:
    """两个站点的关联分析结果。"""

    station_a: str
    station_b: str
    correlations: Dict[str, float] = field(default_factory=dict)  # 各指标皮尔逊相关系数
    mean_correlation: float = 0.0
    lag_correlation: Optional[Dict[str, float]] = None  # 滞后相关（上下游传递）


@dataclass
class SpatialCluster:
    """空间聚类结果。"""

    cluster_id: int
    stations: List[str] = field(default_factory=list)
    centroid: Dict[str, float] = field(default_factory=dict)  # 聚类中心


@dataclass
class SpatialReport:
    """空间关联分析综合报告。"""

    station_profiles: Dict[str, StationProfile] = field(default_factory=dict)
    station_pairs: List[StationPair] = field(default_factory=list)
    correlation_matrix: Optional[pd.DataFrame] = None  # 站点间相关矩阵
    clusters: List[SpatialCluster] = field(default_factory=list)
    heterogeneity_index: float = 0.0  # 空间异质性指数 (0-1)
    summary: str = ""


# ---------------------------------------------------------------------------
# 站点统计
# ---------------------------------------------------------------------------


def _build_station_profile(
    df_station: pd.DataFrame,
    station: str,
    indicators: List[str],
) -> StationProfile:
    """构建单站点概况。"""
    profile = StationProfile(station=station, record_count=len(df_station))

    # 各指标的最差等级计数
    from collections import Counter
    level_counter: Counter = Counter()

    for ind in indicators:
        if ind not in df_station.columns:
            continue
        series = df_station[ind].dropna()
        if len(series) == 0:
            continue
        values = series.to_numpy(dtype=float)
        profile.indicator_means[ind] = float(np.mean(values))
        profile.indicator_stds[ind] = float(np.std(values))

        for v in values:
            lv = _level_for_indicator(ind, float(v))
            level_counter[lv] += 1

    if level_counter:
        profile.dominant_level = level_counter.most_common(1)[0][0]

    return profile


# ---------------------------------------------------------------------------
# 站点间相关性
# ---------------------------------------------------------------------------


def _compute_pairwise_correlation(
    df: pd.DataFrame,
    station_col: str,
    stations: List[str],
    indicators: List[str],
    time_col: Optional[str] = None,
) -> List[StationPair]:
    """计算站点对之间各指标的皮尔逊相关系数。"""
    pairs: List[StationPair] = []

    for i, sta in enumerate(stations):
        for j in range(i + 1, len(stations)):
            stb = stations[j]
            df_a = df[df[station_col] == sta]
            df_b = df[df[station_col] == stb]

            correlations: Dict[str, float] = {}

            if time_col and time_col in df.columns:
                # 按时间对齐后计算相关性
                df_a_indexed = df_a.set_index(time_col)
                df_b_indexed = df_b.set_index(time_col)
                common_times = df_a_indexed.index.intersection(df_b_indexed.index)

                if len(common_times) < 5:
                    continue

                for ind in indicators:
                    if ind not in df_a.columns or ind not in df_b.columns:
                        continue
                    a_vals = df_a_indexed.loc[common_times, ind].dropna()
                    b_vals = df_b_indexed.loc[common_times, ind].dropna()
                    common = a_vals.index.intersection(b_vals.index)
                    if len(common) >= 5:
                        corr = float(np.corrcoef(
                            a_vals.loc[common].to_numpy(dtype=float),
                            b_vals.loc[common].to_numpy(dtype=float),
                        )[0, 1])
                        if not np.isnan(corr):
                            correlations[ind] = round(corr, 4)
            else:
                # 无时间列时，按行号截取相同长度
                min_len = min(len(df_a), len(df_b))
                if min_len < 5:
                    continue
                for ind in indicators:
                    if ind not in df_a.columns or ind not in df_b.columns:
                        continue
                    a_vals = df_a[ind].dropna().to_numpy(dtype=float)[:min_len]
                    b_vals = df_b[ind].dropna().to_numpy(dtype=float)[:min_len]
                    n = min(len(a_vals), len(b_vals))
                    if n >= 5:
                        corr = float(np.corrcoef(a_vals[:n], b_vals[:n])[0, 1])
                        if not np.isnan(corr):
                            correlations[ind] = round(corr, 4)

            if correlations:
                mean_corr = float(np.mean(list(correlations.values())))
                pairs.append(StationPair(
                    station_a=sta,
                    station_b=stb,
                    correlations=correlations,
                    mean_correlation=round(mean_corr, 4),
                ))

    return pairs


# ---------------------------------------------------------------------------
# 简单 K-Means 聚类
# ---------------------------------------------------------------------------


def _cluster_stations(
    profiles: Dict[str, StationProfile],
    indicators: List[str],
    n_clusters: int = 3,
) -> List[SpatialCluster]:
    """基于站点均值特征进行 K-Means 聚类。"""
    stations = list(profiles.keys())
    if len(stations) < n_clusters:
        return [SpatialCluster(cluster_id=0, stations=stations)]

    # 构建特征矩阵
    feature_matrix: List[List[float]] = []
    valid_stations: List[str] = []
    for sta in stations:
        p = profiles[sta]
        row = [p.indicator_means.get(ind, 0.0) for ind in indicators]
        if any(v != 0.0 for v in row):
            feature_matrix.append(row)
            valid_stations.append(sta)

    if len(valid_stations) < n_clusters:
        return [SpatialCluster(cluster_id=0, stations=valid_stations)]

    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        X = StandardScaler().fit_transform(np.array(feature_matrix))
        km = KMeans(n_clusters=min(n_clusters, len(valid_stations)), random_state=42, n_init=10)
        labels = km.fit_predict(X)

        clusters_dict: Dict[int, List[str]] = {}
        for sta, label in zip(valid_stations, labels):
            clusters_dict.setdefault(int(label), []).append(sta)

        clusters: List[SpatialCluster] = []
        for cid, stas in sorted(clusters_dict.items()):
            centroid = {}
            for ind in indicators:
                vals = [profiles[s].indicator_means.get(ind, 0.0) for s in stas]
                centroid[ind] = round(float(np.mean(vals)), 4) if vals else 0.0
            clusters.append(SpatialCluster(
                cluster_id=cid, stations=stas, centroid=centroid,
            ))
        return clusters

    except ImportError:
        return [SpatialCluster(cluster_id=0, stations=valid_stations)]


# ---------------------------------------------------------------------------
# 空间异质性
# ---------------------------------------------------------------------------


def _compute_heterogeneity(profiles: Dict[str, StationProfile], indicators: List[str]) -> float:
    """计算空间异质性指数（各指标变异系数的均值，归一化到 0-1）。"""
    cvs: List[float] = []
    for ind in indicators:
        means = [p.indicator_means.get(ind, 0.0) for p in profiles.values() if ind in p.indicator_means]
        if len(means) < 2:
            continue
        mean_val = float(np.mean(means))
        std_val = float(np.std(means))
        if mean_val != 0:
            cvs.append(abs(std_val / mean_val))

    if not cvs:
        return 0.0
    raw = float(np.mean(cvs))
    return min(1.0, raw)  # 截断到 [0, 1]


# ---------------------------------------------------------------------------
# 主分析函数
# ---------------------------------------------------------------------------


def analyze_spatial(
    df: pd.DataFrame,
    station_col: str = "station",
    time_col: Optional[str] = "date",
    indicators: Optional[List[str]] = None,
    n_clusters: int = 3,
) -> SpatialReport:
    """水质空间关联综合分析。

    Parameters
    ----------
    df : DataFrame
        水质监测数据，需包含站点列。
    station_col : str
        站点标识列名。
    time_col : str, optional
        时间列名（用于时间对齐的相关性计算）。
    indicators : list, optional
        要分析的指标列表。
    n_clusters : int
        聚类数量。

    Returns
    -------
    SpatialReport
    """
    if indicators is None:
        indicators = [c for c in INDICATORS if c in df.columns]

    if station_col not in df.columns:
        return SpatialReport(summary=f"数据中未找到站点列 '{station_col}'，无法进行空间分析。")

    stations = sorted(df[station_col].dropna().unique().tolist())
    if len(stations) < 2:
        return SpatialReport(summary="站点数少于2个，无法进行空间关联分析。")

    # 1. 站点概况
    profiles: Dict[str, StationProfile] = {}
    for sta in stations:
        df_sta: pd.DataFrame = df[df[station_col] == sta]  # type: ignore[assignment]
        profiles[sta] = _build_station_profile(df_sta, str(sta), indicators)

    # 2. 站点间相关性
    pairs = _compute_pairwise_correlation(df, station_col, stations, indicators, time_col)

    # 3. 站点间相关矩阵（基于各指标均值的综合相关）
    corr_matrix = _build_correlation_matrix(pairs, stations)

    # 4. 聚类
    clusters = _cluster_stations(profiles, indicators, n_clusters)

    # 5. 空间异质性
    heterogeneity = _compute_heterogeneity(profiles, indicators)

    report = SpatialReport(
        station_profiles=profiles,
        station_pairs=pairs,
        correlation_matrix=corr_matrix,
        clusters=clusters,
        heterogeneity_index=round(heterogeneity, 4),
    )
    report.summary = _build_summary(report, indicators)
    return report


def _build_correlation_matrix(
    pairs: List[StationPair],
    stations: List[str],
) -> pd.DataFrame:
    """从站点对的结果构建站点间相关矩阵。"""
    n = len(stations)
    mat = np.eye(n)
    sta_idx = {s: i for i, s in enumerate(stations)}

    for pair in pairs:
        i = sta_idx.get(pair.station_a)
        j = sta_idx.get(pair.station_b)
        if i is not None and j is not None:
            mat[i, j] = pair.mean_correlation
            mat[j, i] = pair.mean_correlation

    return pd.DataFrame(mat, index=stations, columns=stations)


def _build_summary(report: SpatialReport, indicators: List[str]) -> str:
    """生成空间分析文字摘要。"""
    lines: List[str] = []
    n_stations = len(report.station_profiles)
    lines.append(f"分析站点数: {n_stations}")
    lines.append(f"空间异质性指数: {report.heterogeneity_index:.3f} "
                 f"({'高' if report.heterogeneity_index > 0.3 else '中' if report.heterogeneity_index > 0.15 else '低'})")

    # 高相关站点对
    high_corr = [p for p in report.station_pairs if p.mean_correlation > 0.7]
    if high_corr:
        lines.append(f"\n高度相关站点对（r > 0.7）: {len(high_corr)} 对")
        for p in sorted(high_corr, key=lambda x: x.mean_correlation, reverse=True)[:5]:
            lines.append(f"  {p.station_a} ↔ {p.station_b}: r={p.mean_correlation:.3f}")

    # 聚类信息
    if report.clusters:
        lines.append(f"\n空间聚类（{len(report.clusters)} 组）:")
        for c in report.clusters:
            lines.append(f"  组 {c.cluster_id}: {', '.join(c.stations)}")

    # 各站点水质概况
    lines.append("\n各站点主要水质等级:")
    for sta, profile in sorted(report.station_profiles.items()):
        means_str = ", ".join(
            f"{ind}={profile.indicator_means.get(ind, 0):.3f}"
            for ind in indicators if ind in profile.indicator_means
        )
        lines.append(f"  {sta}: {profile.dominant_level}类 ({means_str})")

    return "\n".join(lines)
