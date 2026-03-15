from __future__ import annotations

import argparse
import html
import json
import logging
import os
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import pandas as pd
from dotenv import load_dotenv

import app as terminal_app
from water_quality.anomaly import detect_anomalies
from water_quality.fusion import compute_wqi
from water_quality.report import FullReport, generate_report
from water_quality.standards import LEVELS, evaluate_level, get_standards
from water_quality.trend import analyze_trend
from water_quality.vectorstore import BaseRetriever, create_retriever, load_documents
from water_quality.warning import ALERT_LABELS, generate_warnings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

INDICATORS = ["ph", "do", "codmn", "nh3n", "tp"]
ALERT_COLORS = {
    "blue": "#4f9eed",
    "yellow": "#d6a223",
    "orange": "#d97706",
    "red": "#b42318",
}


@dataclass
class DashboardService:
    data_dir: Path
    knowledge_dir: Path
    data_file: Optional[Path]
    top_k: int
    model: str
    vector_backend: str
    persist_dir: Optional[str]
    include_tabular_in_rag: bool

    df: Optional[pd.DataFrame] = None
    data_source: Optional[Path] = None
    retriever: Optional[BaseRetriever] = None
    knowledge_chunks: int = 0
    llm_ready: bool = False

    def refresh(self) -> None:
        load_dotenv()
        self.llm_ready = bool(os.getenv("DEEPSEEK_API_KEY", "").strip())
        self.retriever = create_retriever(
            backend=self.vector_backend,
            persist_dir=self.persist_dir,
        )
        chunks = load_documents(self.knowledge_dir, include_tabular=self.include_tabular_in_rag)
        if chunks:
            try:
                self.retriever.add_chunks(chunks)
                self.knowledge_chunks = self.retriever.count()
            except Exception as exc:
                logger.warning("向量检索初始化失败，回退到 TF-IDF: %s", exc)
                self.retriever = create_retriever(backend="tfidf")
                self.retriever.add_chunks(chunks)
                self.knowledge_chunks = self.retriever.count()
        else:
            self.knowledge_chunks = 0

        self.df, self.data_source = self._load_water_data_with_source()

    def _load_water_data_with_source(self) -> tuple[Optional[pd.DataFrame], Optional[Path]]:
        if self.data_file is not None:
            df = terminal_app._standardize_water_columns(terminal_app._read_table_file(self.data_file))
            return df, self.data_file

        candidates = [
            file_path
            for file_path in sorted(self.data_dir.rglob("*"))
            if file_path.is_file() and file_path.suffix.lower() in {".csv", ".xlsx", ".xls"}
        ]

        best_df: Optional[pd.DataFrame] = None
        best_file: Optional[Path] = None
        best_score = -1
        for file_path in candidates:
            try:
                df = terminal_app._standardize_water_columns(terminal_app._read_table_file(file_path))
            except Exception as exc:
                logger.warning("加载数据失败 %s: %s", file_path, exc)
                continue

            score = terminal_app._water_data_score(df)
            if score > best_score:
                best_df = df
                best_file = file_path
                best_score = score

        if best_df is None or best_file is None:
            return None, None
        if not terminal_app._looks_like_water_data(best_df):
            return None, None
        return best_df, best_file

    def build_context(self) -> dict[str, Any]:
        context: dict[str, Any] = {
            "data_dir": str(self.data_dir),
            "knowledge_dir": str(self.knowledge_dir),
            "data_source": str(self.data_source) if self.data_source else "未找到结构化水质数据",
            "knowledge_chunks": self.knowledge_chunks,
            "llm_ready": self.llm_ready,
            "llm_label": "已配置" if self.llm_ready else "未配置（问答区需设置 DEEPSEEK_API_KEY）",
            "records": 0,
            "columns": 0,
            "latest_values": {},
            "latest_eval": None,
            "latest_wqi": None,
            "trend_report": None,
            "anomaly_report": None,
            "warning_report": None,
            "report": None,
            "table_html": "<p class='empty'>当前没有可展示的数据表。</p>",
            "standards_html": self._render_standards(),
        }

        if self.df is None or self.df.empty:
            return context

        df = self.df
        context["records"] = len(df)
        context["columns"] = len(df.columns)
        context["table_html"] = self._render_table(df.tail(12))

        latest_values = self._latest_indicator_values(df)
        context["latest_values"] = latest_values

        if latest_values:
            latest_eval = evaluate_level(latest_values)
            latest_wqi = compute_wqi(latest_values)
            warning_report = generate_warnings(latest_values, history_df=df)
            context["latest_eval"] = latest_eval
            context["latest_wqi"] = latest_wqi
            context["warning_report"] = warning_report

        time_col = terminal_app._find_time_col(df)
        if time_col:
            try:
                context["trend_report"] = analyze_trend(df, time_col=time_col)
            except Exception as exc:
                logger.warning("趋势分析失败: %s", exc)

        try:
            context["anomaly_report"] = detect_anomalies(df)
        except Exception as exc:
            logger.warning("异常分析失败: %s", exc)

        try:
            full_report = generate_report(df)
            assert isinstance(full_report, FullReport)
            context["report"] = full_report
        except Exception as exc:
            logger.warning("生成报告失败: %s", exc)

        return context

    def answer_question(self, question: str) -> dict[str, Any]:
        question = question.strip()
        if not question:
            return {"ok": False, "error": "请输入问题后再提交。"}

        if self.retriever is None:
            self.refresh()

        assert self.retriever is not None

        try:
            intent = self._detect_intent(question)
            answer = terminal_app.route_question(
                question=question,
                intent=intent,
                retriever=self.retriever,
                df=self.df,
                model=self.model,
                top_k=self.top_k,
            )
            return {"ok": True, "intent": intent, "answer": answer}
        except RuntimeError as exc:
            return {"ok": False, "error": str(exc)}
        except Exception as exc:
            logger.exception("问答失败")
            return {"ok": False, "error": f"处理问题时发生错误: {exc}"}

    def _detect_intent(self, question: str) -> str:
        intent = terminal_app.detect_intent(question)
        if intent != "rag":
            return intent

        normalized = question.replace(" ", "")
        if "水质" in normalized and any(token in normalized for token in ["最新", "当前", "整体", "总体", "情况"]):
            return "evaluate"
        if any(token in normalized for token in ["风险", "告警", "预警"]):
            return "forecast"
        return intent

    def _latest_indicator_values(self, df: pd.DataFrame) -> dict[str, float]:
        available = [column for column in INDICATORS if column in df.columns]
        if not available:
            return {}
        valid_rows = df[available].dropna()
        if valid_rows.empty:
            return {}
        latest = valid_rows.iloc[-1]
        return {indicator: float(latest[indicator]) for indicator in available}

    def render_page(self) -> str:
        context = self.build_context()
        latest_eval = context["latest_eval"]
        latest_wqi = context["latest_wqi"]
        warning_report = context["warning_report"]
        trend_report = context["trend_report"]
        anomaly_report = context["anomaly_report"]
        full_report = context["report"]

        latest_level = "暂无"
        if latest_eval:
            latest_level = str(latest_eval.get("overall_level", "暂无"))

        wqi_label = "暂无"
        if latest_wqi:
            wqi_label = f"{latest_wqi.wqi} / {latest_wqi.level} 类"

        highest_warning = "暂无"
        if warning_report:
            highest_warning = ALERT_LABELS.get(warning_report.highest_level, warning_report.highest_level)

        html_parts = [
            "<!doctype html>",
            "<html lang='zh-CN'>",
            "<head>",
            "  <meta charset='utf-8' />",
            "  <meta name='viewport' content='width=device-width, initial-scale=1' />",
            "  <title>水环境数据分析平台</title>",
            self._style_block(),
            "</head>",
            "<body>",
            "  <div class='page-shell'>",
            "    <header class='hero'>",
            "      <div class='hero-copy'>",
            "        <p class='eyebrow'>QIANTANG RIVER WATER INTELLIGENCE</p>",
            "        <h1>水环境数据分析可视化平台</h1>",
            "        <p class='hero-desc'>从终端版升级为浏览器可视化工作台，支持查看数据、运行分析、预览报告和在线提问。</p>",
            "      </div>",
            "      <div class='hero-meta'>",
            f"        <div><span>数据源</span><strong>{html.escape(context['data_source'])}</strong></div>",
            f"        <div><span>知识库</span><strong>{html.escape(str(context['knowledge_dir']))}</strong></div>",
            f"        <div><span>LLM 状态</span><strong>{html.escape(context['llm_label'])}</strong></div>",
            "      </div>",
            "    </header>",
            "    <section class='metrics-grid'>",
            self._metric_card("数据记录", str(context["records"]), f"{context['columns']} 列"),
            self._metric_card("知识片段", str(context["knowledge_chunks"]), "用于检索问答"),
            self._metric_card("最新等级", latest_level, "单因子最差因子法"),
            self._metric_card("最新 WQI", wqi_label, "综合水质指数"),
            self._metric_card("当前预警", highest_warning, "综合预警结论"),
            self._metric_card("运行入口", "浏览器", "不再只能在终端交互"),
            "    </section>",
            "    <main class='content-grid'>",
            "      <section class='panel wide'>",
            "        <div class='panel-head'><h2>数据概览</h2><button id='refreshButton' class='ghost-btn'>刷新页面</button></div>",
            f"        <div class='table-wrap'>{context['table_html']}</div>",
            "      </section>",
            "      <section class='panel'>",
            "        <div class='panel-head'><h2>最新指标</h2></div>",
            self._render_latest_values(context["latest_values"]),
            "      </section>",
            "      <section class='panel'>",
            "        <div class='panel-head'><h2>标准参考</h2></div>",
            context["standards_html"],
            "      </section>",
            "      <section class='panel wide'>",
            "        <div class='panel-head'><h2>趋势分析</h2></div>",
            self._render_trends(trend_report),
            "      </section>",
            "      <section class='panel'>",
            "        <div class='panel-head'><h2>异常检测</h2></div>",
            self._render_anomalies(anomaly_report),
            "      </section>",
            "      <section class='panel'>",
            "        <div class='panel-head'><h2>预警信息</h2></div>",
            self._render_warnings(warning_report),
            "      </section>",
            "      <section class='panel wide'>",
            "        <div class='panel-head'><h2>报告预览</h2></div>",
            self._render_report(full_report),
            "      </section>",
            "      <section class='panel wide'>",
            "        <div class='panel-head'><h2>智能问答</h2></div>",
            "        <form id='qaForm' class='qa-form'>",
            "          <textarea id='questionInput' name='question' rows='4' placeholder='例如：最近水质整体情况怎么样？有哪些异常指标？'></textarea>",
            "          <div class='qa-actions'>",
            "            <button type='submit' class='primary-btn'>提交问题</button>",
            "            <span class='tip'>支持水质评价、趋势分析、异常检测、预警和知识库问答。</span>",
            "          </div>",
            "        </form>",
            "        <div id='qaResult' class='qa-result empty'>等待你的问题。</div>",
            "      </section>",
            "    </main>",
            "  </div>",
            self._script_block(),
            "</body>",
            "</html>",
        ]
        return "\n".join(html_parts)

    def _metric_card(self, title: str, value: str, detail: str) -> str:
        return (
            "<article class='metric-card'>"
            f"<span>{html.escape(title)}</span>"
            f"<strong>{html.escape(value)}</strong>"
            f"<p>{html.escape(detail)}</p>"
            "</article>"
        )

    def _render_latest_values(self, values: dict[str, float]) -> str:
        if not values:
            return "<p class='empty'>当前数据中没有完整的最新指标记录。</p>"
        items = []
        for indicator, value in values.items():
            items.append(
                "<div class='kv-row'>"
                f"<span>{html.escape(indicator.upper())}</span>"
                f"<strong>{value:.4f}</strong>"
                "</div>"
            )
        return "<div class='kv-list'>" + "".join(items) + "</div>"

    def _render_trends(self, trend_report: Any) -> str:
        if trend_report is None or not trend_report.trends:
            return "<p class='empty'>缺少可用时间列或样本不足，暂时无法展示趋势分析。</p>"

        cards: list[str] = ["<div class='trend-grid'>"]
        for indicator, trend in trend_report.trends.items():
            sparkline = ""
            if self.df is not None and indicator in self.df.columns:
                sparkline = self._sparkline_svg(self.df[indicator].dropna().tail(40).tolist())
            cards.append(
                "<article class='trend-card'>"
                f"<h3>{html.escape(indicator.upper())}</h3>"
                f"{sparkline}"
                f"<p>趋势: <strong>{html.escape(trend.trend)}</strong></p>"
                f"<p>斜率: {trend.slope:.6f}</p>"
                f"<p>变化率: {trend.change_rate:.1f}%</p>"
                f"<p>p 值: {trend.p_value:.4f}</p>"
                "</article>"
            )
        cards.append("</div>")
        cards.append(f"<p class='summary'>{html.escape(trend_report.summary)}</p>")
        return "".join(cards)

    def _render_anomalies(self, anomaly_report: Any) -> str:
        if anomaly_report is None:
            return "<p class='empty'>暂时无法生成异常分析结果。</p>"

        parts = [
            f"<p class='summary'>总记录 {anomaly_report.total_records} 条，识别到 {anomaly_report.anomaly_count} 条异常。</p>"
        ]
        if anomaly_report.summary:
            parts.append("<div class='chip-wrap'>")
            for indicator, count in anomaly_report.summary.items():
                parts.append(f"<span class='chip'>{html.escape(indicator)}: {count}</span>")
            parts.append("</div>")

        if anomaly_report.anomalies:
            parts.append("<ul class='event-list'>")
            for item in anomaly_report.anomalies[:8]:
                parts.append(
                    "<li>"
                    f"<strong>{html.escape(item.indicator)}</strong> "
                    f"第 {item.index} 行，值 {item.value:.4f}，{html.escape(item.reason)}"
                    "</li>"
                )
            parts.append("</ul>")
        return "".join(parts)

    def _render_warnings(self, warning_report: Any) -> str:
        if warning_report is None:
            return "<p class='empty'>没有足够数据生成预警。</p>"

        label = ALERT_LABELS.get(warning_report.highest_level, warning_report.highest_level)
        color = ALERT_COLORS.get(warning_report.highest_level, "#4f9eed")
        parts = [
            f"<div class='alert-banner' style='--alert:{color}'>{html.escape(label)}</div>"
        ]
        if not warning_report.alerts:
            parts.append("<p class='summary'>所有指标正常，无预警。</p>")
            return "".join(parts)

        parts.append("<ul class='event-list'>")
        for alert in warning_report.alerts[:8]:
            parts.append(f"<li>{html.escape(alert.message)}</li>")
        parts.append("</ul>")
        return "".join(parts)

    def _render_report(self, report: Optional[FullReport]) -> str:
        if report is None or not report.sections:
            return "<p class='empty'>报告生成失败或当前没有可分析的数据。</p>"

        sections = ["<div class='report-grid'>"]
        for section in report.sections:
            sections.append(
                "<article class='report-card'>"
                f"<h3>{html.escape(section.title)}</h3>"
                f"<pre>{html.escape(section.content)}</pre>"
                "</article>"
            )
        sections.append("</div>")
        return "".join(sections)

    def _render_standards(self) -> str:
        standards = get_standards()
        rows = [
            "<table class='data-table standards-table'><thead><tr><th>指标</th><th>说明</th></tr></thead><tbody>"
        ]
        for indicator in INDICATORS:
            rule = standards.get(indicator, {})
            if "range" in rule:
                desc = f"范围 {rule['range']['min']} - {rule['range']['max']}"
            else:
                thresholds = rule.get("thresholds", {})
                desc = " / ".join(f"{level}:{thresholds[level]}" for level in LEVELS if level in thresholds)
            rows.append(
                f"<tr><td>{html.escape(indicator.upper())}</td><td>{html.escape(desc)}</td></tr>"
            )
        rows.append("</tbody></table>")
        return "".join(rows)

    def _render_table(self, df: pd.DataFrame) -> str:
        show_df = df.copy()
        for column in show_df.columns:
            if pd.api.types.is_float_dtype(show_df[column]):
                show_df[column] = show_df[column].map(lambda value: round(float(value), 4) if pd.notna(value) else value)
        return show_df.to_html(index=False, classes="data-table", border=0)

    def _sparkline_svg(self, values: list[float]) -> str:
        if len(values) < 2:
            return ""
        width = 220
        height = 72
        padding = 6
        minimum = min(values)
        maximum = max(values)
        span = maximum - minimum or 1.0
        points: list[str] = []
        for idx, value in enumerate(values):
            x = padding + (width - 2 * padding) * idx / (len(values) - 1)
            y = height - padding - (height - 2 * padding) * ((value - minimum) / span)
            points.append(f"{x:.2f},{y:.2f}")
        polyline = " ".join(points)
        return (
            "<svg class='sparkline' viewBox='0 0 220 72' preserveAspectRatio='none'>"
            "<defs><linearGradient id='sparkFill' x1='0' y1='0' x2='0' y2='1'>"
            "<stop offset='0%' stop-color='#0f766e' stop-opacity='0.28'></stop>"
            "<stop offset='100%' stop-color='#0f766e' stop-opacity='0'></stop>"
            "</linearGradient></defs>"
            f"<polyline points='{polyline}' fill='none' stroke='#0f766e' stroke-width='2.4' stroke-linecap='round' stroke-linejoin='round'></polyline>"
            "</svg>"
        )

    def _style_block(self) -> str:
        return """
<style>
:root {
  --bg: #eef5ef;
  --panel: rgba(255, 255, 255, 0.88);
  --panel-strong: #ffffff;
  --line: rgba(15, 23, 42, 0.08);
  --ink: #10231d;
  --muted: #5d746d;
  --accent: #0f766e;
  --accent-soft: #d8efe8;
  --warm: #f2e8d7;
  --shadow: 0 18px 50px rgba(16, 35, 29, 0.08);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  color: var(--ink);
  font-family: "Avenir Next", "PingFang SC", "Hiragino Sans GB", "Source Han Sans SC", sans-serif;
  background:
    radial-gradient(circle at top left, rgba(15, 118, 110, 0.18), transparent 28%),
    radial-gradient(circle at right center, rgba(210, 170, 88, 0.18), transparent 24%),
    linear-gradient(180deg, #f7fbf7 0%, var(--bg) 100%);
}
.page-shell {
  max-width: 1380px;
  margin: 0 auto;
  padding: 28px;
}
.hero {
  display: grid;
  grid-template-columns: 1.5fr 1fr;
  gap: 24px;
  margin-bottom: 24px;
  padding: 32px;
  border: 1px solid var(--line);
  border-radius: 28px;
  background: linear-gradient(135deg, rgba(255,255,255,0.96), rgba(239,248,244,0.92));
  box-shadow: var(--shadow);
}
.eyebrow {
  margin: 0 0 12px;
  letter-spacing: 0.22em;
  font-size: 12px;
  color: var(--accent);
}
h1, h2, h3, strong {
  font-family: "Iowan Old Style", "Palatino Linotype", "Source Han Serif SC", serif;
}
h1 {
  margin: 0;
  font-size: clamp(32px, 4vw, 52px);
  line-height: 1.06;
}
.hero-desc {
  max-width: 720px;
  margin: 14px 0 0;
  font-size: 16px;
  line-height: 1.7;
  color: var(--muted);
}
.hero-meta {
  display: grid;
  gap: 14px;
  align-content: start;
}
.hero-meta div {
  padding: 16px 18px;
  border-radius: 18px;
  border: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.76);
}
.hero-meta span {
  display: block;
  margin-bottom: 8px;
  color: var(--muted);
  font-size: 13px;
}
.hero-meta strong {
  font-size: 16px;
  line-height: 1.4;
}
.metrics-grid {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 16px;
  margin-bottom: 24px;
}
.metric-card, .panel {
  border: 1px solid var(--line);
  background: var(--panel);
  border-radius: 24px;
  box-shadow: var(--shadow);
  backdrop-filter: blur(10px);
}
.metric-card {
  padding: 18px;
}
.metric-card span {
  display: block;
  margin-bottom: 14px;
  color: var(--muted);
  font-size: 13px;
}
.metric-card strong {
  display: block;
  font-size: 26px;
  line-height: 1.2;
}
.metric-card p {
  margin: 10px 0 0;
  color: var(--muted);
  font-size: 13px;
  line-height: 1.5;
}
.content-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 18px;
}
.panel {
  padding: 22px;
}
.panel.wide {
  grid-column: 1 / -1;
}
.panel-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  margin-bottom: 16px;
}
.panel-head h2 {
  margin: 0;
  font-size: 26px;
}
.ghost-btn, .primary-btn {
  border: 0;
  border-radius: 999px;
  cursor: pointer;
  font: inherit;
}
.ghost-btn {
  padding: 10px 14px;
  background: var(--accent-soft);
  color: var(--accent);
}
.primary-btn {
  padding: 12px 18px;
  background: linear-gradient(135deg, #0f766e, #115e59);
  color: #fff;
}
.table-wrap {
  overflow: auto;
}
.data-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
}
.data-table th, .data-table td {
  padding: 12px 10px;
  border-bottom: 1px solid var(--line);
  text-align: left;
}
.data-table th {
  color: var(--muted);
  font-weight: 600;
}
.kv-list {
  display: grid;
  gap: 12px;
}
.kv-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 14px;
  border-radius: 16px;
  background: rgba(216, 239, 232, 0.55);
}
.trend-grid, .report-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}
.trend-card, .report-card {
  padding: 16px;
  border-radius: 20px;
  background: var(--panel-strong);
  border: 1px solid var(--line);
}
.trend-card h3, .report-card h3 {
  margin: 0 0 10px;
  font-size: 22px;
}
.trend-card p {
  margin: 6px 0 0;
  color: var(--muted);
}
.sparkline {
  display: block;
  width: 100%;
  height: 72px;
  margin: 8px 0 12px;
  border-radius: 14px;
  background: linear-gradient(180deg, rgba(15, 118, 110, 0.08), rgba(15, 118, 110, 0.01));
}
.summary {
  margin: 14px 0 0;
  color: var(--muted);
  line-height: 1.7;
}
.chip-wrap {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin: 14px 0;
}
.chip {
  padding: 8px 12px;
  border-radius: 999px;
  background: #edf4f1;
  color: #21443d;
  font-size: 13px;
}
.event-list {
  margin: 14px 0 0;
  padding-left: 18px;
  color: var(--muted);
  line-height: 1.75;
}
.alert-banner {
  padding: 14px 16px;
  border-radius: 18px;
  background: color-mix(in srgb, var(--alert) 14%, white);
  color: var(--alert);
  font-weight: 700;
}
.report-card pre {
  margin: 0;
  white-space: pre-wrap;
  word-break: break-word;
  color: var(--muted);
  line-height: 1.65;
  font-family: "SFMono-Regular", "JetBrains Mono", monospace;
  font-size: 13px;
}
.qa-form textarea {
  width: 100%;
  resize: vertical;
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 16px;
  font: inherit;
  background: rgba(255,255,255,0.92);
  min-height: 120px;
}
.qa-actions {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
  margin-top: 14px;
}
.tip, .empty {
  color: var(--muted);
}
.qa-result {
  margin-top: 18px;
  padding: 18px;
  border-radius: 18px;
  border: 1px solid var(--line);
  background: rgba(255,255,255,0.74);
  white-space: pre-wrap;
  line-height: 1.72;
}
@media (max-width: 1100px) {
  .hero, .content-grid {
    grid-template-columns: 1fr;
  }
  .metrics-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
  .trend-grid, .report-grid {
    grid-template-columns: 1fr;
  }
}
@media (max-width: 680px) {
  .page-shell {
    padding: 16px;
  }
  .hero, .panel {
    padding: 18px;
    border-radius: 20px;
  }
  .metrics-grid {
    grid-template-columns: 1fr;
  }
  .panel-head {
    flex-direction: column;
    align-items: flex-start;
  }
  .qa-actions {
    flex-direction: column;
    align-items: flex-start;
  }
}
</style>
"""

    def _script_block(self) -> str:
        return """
<script>
const form = document.getElementById("qaForm");
const result = document.getElementById("qaResult");
const refreshButton = document.getElementById("refreshButton");

refreshButton.addEventListener("click", () => window.location.reload());

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = document.getElementById("questionInput").value.trim();
  if (!question) {
    result.textContent = "请输入问题后再提交。";
    result.classList.remove("empty");
    return;
  }
  result.textContent = "正在分析，请稍候...";
  result.classList.remove("empty");

  try {
    const response = await fetch("/api/question", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({question})
    });
    const payload = await response.json();
    if (!payload.ok) {
      result.textContent = payload.error || "请求失败。";
      return;
    }
    result.textContent = "意图: " + payload.intent + "\\n\\n" + payload.answer;
  } catch (error) {
    result.textContent = "请求失败: " + error;
  }
});
</script>
"""


class DashboardHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], service: DashboardService) -> None:
        super().__init__(server_address, DashboardHandler)
        self.service = service


class DashboardHandler(BaseHTTPRequestHandler):
    server: DashboardHTTPServer

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(self.server.service.render_page())
            return
        if parsed.path == "/api/dashboard":
            payload = self.server.service.build_context()
            serialized = self._serialize_dashboard(payload)
            self._send_json(serialized)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/question":
            data = self._read_json_body()
            question = str(data.get("question", ""))
            payload = self.server.service.answer_question(question)
            status = HTTPStatus.OK if payload.get("ok") else HTTPStatus.BAD_REQUEST
            self._send_json(payload, status=status)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, format: str, *args: Any) -> None:
        logger.info("%s - %s", self.address_string(), format % args)

    def _read_json_body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def _send_html(self, body: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serialize_dashboard(self, payload: dict[str, Any]) -> dict[str, Any]:
        latest_eval = payload.get("latest_eval")
        latest_wqi = payload.get("latest_wqi")
        warning_report = payload.get("warning_report")
        anomaly_report = payload.get("anomaly_report")
        return {
            "records": payload.get("records"),
            "columns": payload.get("columns"),
            "data_source": payload.get("data_source"),
            "knowledge_chunks": payload.get("knowledge_chunks"),
            "llm_ready": payload.get("llm_ready"),
            "latest_values": payload.get("latest_values"),
            "latest_eval": latest_eval,
            "latest_wqi": {
                "wqi": latest_wqi.wqi,
                "level": latest_wqi.level,
                "indicator_scores": latest_wqi.indicator_scores,
            } if latest_wqi else None,
            "warning_summary": warning_report.summary if warning_report else None,
            "anomaly_count": anomaly_report.anomaly_count if anomaly_report else None,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="水环境数据分析本地 Web 平台")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--data-dir", default="data/datasets", help="结构化数据目录")
    parser.add_argument("--data-file", default=None, help="显式指定结构化水质数据文件")
    parser.add_argument("--knowledge-dir", default="data/knowledge", help="知识文档目录")
    parser.add_argument("--top-k", type=int, default=4, help="RAG 检索数量")
    parser.add_argument(
        "--model",
        default=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        help="LLM 模型名称",
    )
    parser.add_argument(
        "--vector-backend",
        choices=["auto", "chroma", "tfidf"],
        default="tfidf",
        help="向量检索后端",
    )
    parser.add_argument("--persist-dir", default=None, help="ChromaDB 持久化目录")
    parser.add_argument(
        "--include-tabular-in-rag",
        action="store_true",
        help="将 CSV/Excel 也纳入 RAG 检索",
    )
    return parser


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()
    service = DashboardService(
        data_dir=Path(args.data_dir),
        knowledge_dir=Path(args.knowledge_dir),
        data_file=Path(args.data_file) if args.data_file else None,
        top_k=args.top_k,
        model=args.model,
        vector_backend=args.vector_backend,
        persist_dir=args.persist_dir,
        include_tabular_in_rag=args.include_tabular_in_rag,
    )
    service.refresh()

    server = DashboardHTTPServer((args.host, args.port), service)
    logger.info("Web 平台已启动: http://%s:%s", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Web 平台已停止")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
