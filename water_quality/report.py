"""水质分析报告生成模块。

支持两种模式：
  1. 模板化报告 — 基于数据统计自动生成 Markdown 报告
  2. LLM 增强报告 — 结合分析结果，由 LLM 生成专业解读和建议

提供结构化输出设计，可输出 Markdown / JSON 格式。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .anomaly import detect_anomalies
from .fusion import batch_fusion
from .trend import analyze_trend

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class ReportSection:
    """报告的一个章节。"""

    title: str
    content: str
    data: Optional[Dict[str, Any]] = None  # 结构化数据（用于 JSON 输出）


@dataclass
class FullReport:
    """完整水质报告。"""

    title: str = "水质分析报告"
    content: str = ""
    sections: List[ReportSection] = field(default_factory=list)
    structured_data: Optional[Dict[str, Any]] = None  # 完整 JSON 数据


# ---------------------------------------------------------------------------
# 保存
# ---------------------------------------------------------------------------


def save_report(report: FullReport, out_path: Path) -> Path:
    """保存报告到文件。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report.content, encoding="utf-8")

    # 同时输出 JSON 格式的结构化数据
    if report.structured_data:
        json_path = out_path.with_suffix(".json")
        json_path.write_text(
            json.dumps(report.structured_data, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("结构化数据已保存: %s", json_path)

    return out_path


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

INDICATORS = ["ph", "do", "codmn", "nh3n", "tp"]


def _find_time_col(df: pd.DataFrame) -> Optional[str]:
    """自动检测时间列。"""
    candidates = ["date", "datetime", "time", "timestamp", "日期", "时间", "监测时间", "采样时间"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _distribution(series: pd.Series) -> pd.DataFrame:
    counts = series.value_counts(dropna=False).reset_index()
    counts.columns = ["level", "count"]
    counts["ratio"] = (counts["count"] / counts["count"].sum()).round(4)
    return counts


def _format_table(df: pd.DataFrame, index: bool = True) -> str:
    """优先输出 Markdown 表格，缺少 tabulate 时退回普通文本。"""
    try:
        return df.to_markdown(index=index)
    except ImportError:
        return df.to_string(index=index)


# ---------------------------------------------------------------------------
# 模板化报告生成
# ---------------------------------------------------------------------------


def generate_report(
    df: pd.DataFrame,
    out_path: Optional[Path] = None,
    title: str = "水质分析报告",
    top_n: int = 5,
    time_col: Optional[str] = None,
    use_llm_commentary: bool = False,
    use_langchain: Optional[bool] = None,
    model_name: Optional[str] = None,
) -> FullReport:
    """生成水质分析报告。

    Parameters
    ----------
    df : DataFrame
        水质监测数据。
    out_path : Path, optional
        输出路径。为 None 时返回 FullReport 对象。
    title : str
        报告标题。
    top_n : int
        最差样本展示数量。
    time_col : str, optional
        时间列名。
    use_llm_commentary : bool
        是否使用 LLM 生成 AI 解读。
    use_langchain : bool, optional
        历史参数名，保留兼容。
    model_name : str, optional
        LLM 模型名称。

    Returns
    -------
    FullReport
    """
    if use_langchain is not None:
        use_llm_commentary = use_llm_commentary or use_langchain

    if time_col is None:
        time_col = _find_time_col(df)

    available = [c for c in INDICATORS if c in df.columns]
    sections: List[ReportSection] = []
    structured: Dict[str, Any] = {"title": title, "total_records": len(df)}

    # ── 1. 基础统计 ──────────────────────────────────────────────
    if available:
        desc_df = df[available].describe()
        section_content = f"数据量: {len(df)} 条\n\n{_format_table(desc_df)}"
        structured["basic_stats"] = json.loads(desc_df.to_json())
        sections.append(ReportSection(
            title="基础统计",
            content=section_content,
            data=structured["basic_stats"],
        ))

    # ── 2. 综合水质评价 ──────────────────────────────────────────
    fusion_report = batch_fusion(df)
    fusion_data = {
        "wqi_mean": fusion_report.wqi_mean,
        "wqi_std": fusion_report.wqi_std,
        "level_distribution": fusion_report.level_distribution,
    }
    structured["fusion"] = fusion_data

    level_lines = [f"- WQI 均值: {fusion_report.wqi_mean}, 标准差: {fusion_report.wqi_std}"]
    if fusion_report.level_distribution:
        level_lines.append("- 等级分布:")
        for lv, cnt in sorted(fusion_report.level_distribution.items()):
            level_lines.append(f"  - {lv}类: {cnt} 条")
    sections.append(ReportSection(
        title="综合水质评价",
        content="\n".join(level_lines),
        data=fusion_data,
    ))

    # ── 3. 趋势分析 ──────────────────────────────────────────────
    if time_col:
        trend_report = analyze_trend(df, time_col=time_col)
        trend_data: Dict[str, Any] = {}
        trend_lines: List[str] = []
        for ind, t in trend_report.trends.items():
            trend_data[ind] = {
                "trend": t.trend, "slope": t.slope, "p_value": t.p_value,
                "mean": t.mean, "change_rate": t.change_rate,
            }
            trend_lines.append(
                f"- {ind}: {t.trend}（斜率={t.slope:.6f}, p={t.p_value:.4f}, "
                f"变化率={t.change_rate:.1f}%）"
            )
        structured["trends"] = trend_data
        sections.append(ReportSection(
            title="趋势分析",
            content="\n".join(trend_lines) if trend_lines else "数据不足，无法分析趋势",
            data=trend_data,
        ))

    # ── 4. 异常检测 ──────────────────────────────────────────────
    anomaly_report = detect_anomalies(df)
    anomaly_data = {
        "anomaly_count": anomaly_report.anomaly_count,
        "by_indicator": anomaly_report.summary,
    }
    structured["anomalies"] = anomaly_data

    anomaly_lines = [f"异常总数: {anomaly_report.anomaly_count}"]
    for ind, cnt in anomaly_report.summary.items():
        anomaly_lines.append(f"- {ind}: {cnt} 处异常")
    sections.append(ReportSection(
        title="异常检测",
        content="\n".join(anomaly_lines),
        data=anomaly_data,
    ))

    # ── 5. 最差样本 ──────────────────────────────────────────────
    if available:
        pollutant_cols = [c for c in ["codmn", "nh3n", "tp"] if c in df.columns]
        if pollutant_cols:
            score = df[pollutant_cols].sum(axis=1)
            worst = df.assign(_score=score).sort_values("_score", ascending=False).head(top_n)
            show_cols = [c for c in available if c in worst.columns]
            sections.append(ReportSection(
                title=f"最差样本（Top {top_n}）",
                content=_format_table(worst[show_cols], index=False),
            ))

    # ── 组装报告 ──────────────────────────────────────────────────
    md_lines: List[str] = [f"# {title}", ""]
    for sec in sections:
        md_lines.append(f"## {sec.title}")
        md_lines.append("")
        md_lines.append(sec.content)
        md_lines.append("")

    report = FullReport(
        title=title,
        content="\n".join(md_lines),
        sections=sections,
        structured_data=structured,
    )

    # ── LLM 增强解读 ──────────────────────────────────────────────
    if use_llm_commentary and model_name:
        ai_commentary = _generate_ai_commentary(structured, model_name)
        if ai_commentary:
            report.content += f"\n## AI 专业解读\n\n{ai_commentary}\n"
            report.sections.append(ReportSection(
                title="AI 专业解读", content=ai_commentary,
            ))

    if out_path is not None:
        save_report(report, out_path)

    return report


# ---------------------------------------------------------------------------
# LLM 增强解读
# ---------------------------------------------------------------------------


def _generate_ai_commentary(
    structured_data: Dict[str, Any],
    model_name: str,
) -> Optional[str]:
    """使用 LLM 对分析结果生成专业解读。"""
    import os

    try:
        from openai import OpenAI

        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            logger.warning("DEEPSEEK_API_KEY 未设置，跳过 AI 解读")
            return None

        base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
        client = OpenAI(api_key=api_key, base_url=base_url)

        context = json.dumps(structured_data, ensure_ascii=False, indent=2, default=str)

        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是水环境分析专家，请根据以下水质分析数据生成专业解读。"
                        "内容应包括：水质总体评价、主要污染因子分析、趋势判断、"
                        "风险提示和管理建议。语言严谨专业，适合写入正式报告。"
                    ),
                },
                {"role": "user", "content": f"分析数据：\n{context}"},
            ],
        )
        return (resp.choices[0].message.content or "").strip()

    except Exception as exc:
        logger.warning("AI 解读生成失败: %s", exc)
        return None
