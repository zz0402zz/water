from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from water_quality.anomaly import detect_anomalies
from water_quality.fusion import batch_fusion, compute_wqi
from water_quality.season import analyze_season
from water_quality.spatial import analyze_spatial
from water_quality.standards import evaluate_level, get_standards
from water_quality.trend import analyze_trend
from water_quality.vectorstore import (
    BaseRetriever,
    Chunk,
    create_retriever,
    load_documents,
)
from water_quality.warning import generate_warnings

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

INDICATOR_COLUMNS = {"ph", "do", "codmn", "nh3n", "tp"}
TIME_COLUMN_CANDIDATES = {
    "date", "datetime", "time", "timestamp", "日期", "时间", "监测时间", "采样时间",
}
STATION_COLUMN_CANDIDATES = {
    "station", "site", "断面", "站点", "监测点", "site_name", "station_name",
}
COLUMN_ALIASES = {
    "ph": "ph",
    "ph值": "ph",
    "p_h": "ph",
    "do": "do",
    "dissolvedoxygen": "do",
    "溶解氧": "do",
    "codmn": "codmn",
    "高锰酸盐指数": "codmn",
    "permanganateindex(codmn)": "codmn",
    "permanganateindex": "codmn",
    "nh3n": "nh3n",
    "nh3-n": "nh3n",
    "ammonianitrogen": "nh3n",
    "氨氮": "nh3n",
    "tp": "tp",
    "totalphosphorus": "tp",
    "总磷": "tp",
}


# ═══════════════════════════════════════════════════════════════════════════
# LLM 客户端
# ═══════════════════════════════════════════════════════════════════════════

_llm_client: Optional[OpenAI] = None


def _get_llm_client() -> OpenAI:
    global _llm_client
    if _llm_client is None:
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY 未设置。请在 .env 文件或环境变量中配置。")
        base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
        _llm_client = OpenAI(api_key=api_key, base_url=base_url)
    return _llm_client


def ask_llm(
    question: str,
    context: str = "",
    model: str = "deepseek-chat",
    system_prompt: Optional[str] = None,
    response_format: Optional[str] = None,
) -> str:
    """调用 LLM 生成回答。

    Parameters
    ----------
    question : str
        用户问题。
    context : str
        检索到的上下文或分析结果。
    model : str
        模型名称。
    system_prompt : str, optional
        自定义系统提示词。
    response_format : str, optional
        设为 "json" 时请求 JSON 格式输出。
    """
    client = _get_llm_client()

    if system_prompt is None:
        system_prompt = (
            "你是一个专业的水环境数据分析助手。"
            "请根据提供的数据分析结果和上下文，用中文回答用户问题。"
            "如果信息不足，请如实说明。回答应专业、准确、条理清晰。"
        )

    user_content = f"问题：\n{question}"
    if context:
        user_content += f"\n\n参考数据/上下文：\n{context}"

    kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }
    if response_format == "json":
        kwargs["response_format"] = {"type": "json_object"}

    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as exc:
        logger.error("LLM 请求失败: %s", exc)
        return f"LLM 请求失败: {exc}"

    return (resp.choices[0].message.content or "").strip()


# ═══════════════════════════════════════════════════════════════════════════
# 意图识别
# ═══════════════════════════════════════════════════════════════════════════

INTENT_RULES = [
    ("evaluate",  ["水质评价", "实时评价", "水质类别", "达标", "超标", "水质等级"]),
    ("extreme",   ["最低", "最小", "最高", "最大", "极值"]),
    ("list",      ["列出", "例举", "所有", "汇总", "统计"]),
    ("trend",     ["趋势", "变化规律", "时序", "季节性", "历史变化", "演变"]),
    ("anomaly",   ["异常", "异常识别", "异常分析", "突变", "离群"]),
    ("season",    ["汛期", "非汛期", "丰水期", "枯水期", "雨季", "旱季"]),
    ("spatial",   ["空间", "站点对比", "上下游", "断面", "区域差异", "空间关联"]),
    ("forecast",  ["预测", "预报", "预警", "风险"]),
    ("report",    ["报告", "生成报告", "总结报告", "分析报告"]),
    ("standard",  ["标准", "国标", "阈值", "限值", "GB"]),
]


def detect_intent(question: str) -> str:
    """基于关键词的意图识别。"""
    q = question.strip()
    if not q:
        return "empty"

    for intent, keywords in INTENT_RULES:
        if any(k in q for k in keywords):
            return intent
    return "rag"


# ═══════════════════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════════════════


def _normalize_label(label: Any) -> str:
    return "".join(str(label).strip().lower().split())


def _standardize_water_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: Dict[str, str] = {}
    existing = {str(c) for c in df.columns}

    for column in df.columns:
        target = COLUMN_ALIASES.get(_normalize_label(column))
        if target and target not in existing:
            rename_map[str(column)] = target
            existing.add(target)

    return df.rename(columns=rename_map) if rename_map else df


def _read_table_file(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        last_error: Optional[Exception] = None
        for encoding in ("utf-8", "utf-8-sig", "gb18030"):
            try:
                return pd.read_csv(file_path, encoding=encoding)
            except Exception as exc:  # pragma: no cover - keeps best-effort loading concise
                last_error = exc
        assert last_error is not None
        raise last_error

    engine = "xlrd" if suffix == ".xls" else "openpyxl"
    return pd.read_excel(file_path, engine=engine)


def _water_data_score(df: pd.DataFrame) -> int:
    normalized = {_normalize_label(c) for c in df.columns}
    indicators = len(normalized & INDICATOR_COLUMNS)
    has_time = bool(normalized & {_normalize_label(c) for c in TIME_COLUMN_CANDIDATES})
    has_station = bool(normalized & {_normalize_label(c) for c in STATION_COLUMN_CANDIDATES})
    return indicators * 3 + (2 if has_time else 0) + (1 if has_station else 0)


def _looks_like_water_data(df: pd.DataFrame) -> bool:
    normalized = {_normalize_label(c) for c in df.columns}
    indicators = len(normalized & INDICATOR_COLUMNS)
    has_time = bool(normalized & {_normalize_label(c) for c in TIME_COLUMN_CANDIDATES})
    return indicators >= 2 or (indicators >= 1 and has_time)


def _load_water_data(data_dir: Path, data_file: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """优先读取显式指定的数据文件，否则自动挑选最像水质监测表的文件。"""
    if data_file is not None:
        df = _standardize_water_columns(_read_table_file(data_file))
        if not _looks_like_water_data(df):
            logger.warning("指定的数据文件 %s 缺少典型水质指标列，分析功能可能受限", data_file)
        return df

    candidates = [
        file_path
        for file_path in sorted(data_dir.rglob("*"))
        if file_path.is_file() and file_path.suffix.lower() in {".csv", ".xlsx", ".xls"}
    ]

    best_df: Optional[pd.DataFrame] = None
    best_file: Optional[Path] = None
    best_score = -1

    for file_path in candidates:
        try:
            df = _standardize_water_columns(_read_table_file(file_path))
        except Exception as exc:
            logger.warning("加载数据失败 %s: %s", file_path, exc)
            continue

        score = _water_data_score(df)
        if score > best_score:
            best_df = df
            best_file = file_path
            best_score = score

    if best_df is None or best_file is None:
        return None

    if not _looks_like_water_data(best_df):
        logger.warning("未发现典型水质监测表，已跳过结构化分析数据加载")
        return None

    logger.info("自动选择结构化水质数据文件: %s", best_file)
    return best_df


# ═══════════════════════════════════════════════════════════════════════════
# 各意图的处理函数
# ═══════════════════════════════════════════════════════════════════════════


def handle_rag(
    question: str,
    retriever: BaseRetriever,
    model: str,
    top_k: int = 4,
    **_: Any,
) -> str:
    """RAG: 检索 + LLM 生成回答。"""
    results = retriever.retrieve(question, top_k=top_k)
    if not results:
        return "未检索到相关内容。请确认知识目录中是否包含可检索的文本资料。"

    context = "\n\n".join(
        f"[来源: {chunk.source} | 相关度={score:.3f}]\n{chunk.text}"
        for chunk, score in results
    )
    return ask_llm(question, context=context, model=model)


def handle_evaluate(
    question: str,
    df: Optional[pd.DataFrame],
    model: str,
    **_: Any,
) -> str:
    """水质实时评价。"""
    if df is None:
        return "未找到水质监测数据，无法进行评价。"

    indicators = ["ph", "do", "codmn", "nh3n", "tp"]
    available = [c for c in indicators if c in df.columns]
    if not available:
        return "数据中未找到标准水质指标列（ph, do, codmn, nh3n, tp）。"

    latest = df[available].dropna().iloc[-1] if len(df) > 0 else None
    if latest is None:
        return "数据为空。"

    values = {ind: float(latest[ind]) for ind in available}
    eval_result = evaluate_level(values)
    wqi_result = compute_wqi(values)

    context = (
        f"水质指标值: {json.dumps(values, ensure_ascii=False)}\n"
        f"单因子评价（最差因子法）: {json.dumps(eval_result, ensure_ascii=False)}\n"
        f"综合水质指数 WQI: {wqi_result.wqi}（{wqi_result.level}类）\n"
        f"各指标得分: {json.dumps(wqi_result.indicator_scores, ensure_ascii=False)}"
    )
    return ask_llm(question, context=context, model=model)


def handle_trend(
    question: str,
    df: Optional[pd.DataFrame],
    model: str,
    **_: Any,
) -> str:
    """历史水质趋势分析。"""
    if df is None:
        return "未找到水质监测数据，无法进行趋势分析。"

    time_col = _find_time_col(df) or "date"
    report = analyze_trend(df, time_col=time_col)

    context_lines = ["=== 趋势分析结果 ==="]
    for ind, t in report.trends.items():
        context_lines.append(
            f"{ind}: {t.trend}（斜率={t.slope:.6f}, p={t.p_value:.4f}, "
            f"均值={t.mean:.4f}, 变化率={t.change_rate:.1f}%）"
        )
    if report.seasonals:
        context_lines.append("\n=== 季节性分析 ===")
        for ind, s in report.seasonals.items():
            context_lines.append(
                f"{ind}: 强度={s.seasonal_strength:.2f}, "
                f"高峰={s.peak_month}月, 低谷={s.trough_month}月"
            )
    context_lines.append(f"\n摘要: {report.summary}")

    return ask_llm(question, context="\n".join(context_lines), model=model)


def handle_anomaly(
    question: str,
    df: Optional[pd.DataFrame],
    model: str,
    **_: Any,
) -> str:
    """异常识别与分析。"""
    if df is None:
        return "未找到水质监测数据，无法进行异常检测。"

    report = detect_anomalies(df)

    context_lines = [
        f"总记录: {report.total_records}, 异常数: {report.anomaly_count}",
        "各指标异常统计:",
    ]
    for ind, count in report.summary.items():
        context_lines.append(f"  {ind}: {count} 处异常")

    if report.anomalies:
        context_lines.append("\n典型异常记录（前15条）:")
        for a in report.anomalies[:15]:
            context_lines.append(
                f"  [{a.severity}] 行{a.index}, {a.indicator}={a.value:.4f} "
                f"({a.method}: {a.reason})"
            )

    return ask_llm(question, context="\n".join(context_lines), model=model)


def handle_season(
    question: str,
    df: Optional[pd.DataFrame],
    model: str,
    **_: Any,
) -> str:
    """汛期/非汛期对比分析。"""
    if df is None:
        return "未找到水质监测数据，无法进行汛期分析。"

    time_col = _find_time_col(df)
    if not time_col:
        return "数据中未找到时间列，无法区分汛期/非汛期。"

    report = analyze_season(df, time_col=time_col)

    context_lines = [report.summary]
    for ind, comp in report.comparisons.items():
        context_lines.append(
            f"\n{ind} 对比: 汛期均值={comp.flood_stats.mean:.4f}, "
            f"非汛期均值={comp.dry_stats.mean:.4f}, "
            f"差异={comp.mean_diff_pct:.1f}%, p={comp.p_value:.4f}"
            f"{' (显著)' if comp.significant else ''}"
        )

    return ask_llm(question, context="\n".join(context_lines), model=model)


def handle_spatial(
    question: str,
    df: Optional[pd.DataFrame],
    model: str,
    **_: Any,
) -> str:
    """空间关联分析。"""
    if df is None:
        return "未找到水质监测数据，无法进行空间分析。"

    station_col = _find_station_col(df)
    if not station_col:
        return "数据中未找到站点列（如 station, site, 断面），无法进行空间分析。"

    time_col = _find_time_col(df)
    report = analyze_spatial(df, station_col=station_col, time_col=time_col)

    return ask_llm(question, context=report.summary, model=model)


def handle_forecast(
    question: str,
    df: Optional[pd.DataFrame],
    model: str,
    **_: Any,
) -> str:
    """水质预警。"""
    if df is None:
        return "未找到水质监测数据，无法生成预警。"

    indicators = ["ph", "do", "codmn", "nh3n", "tp"]
    available = [c for c in indicators if c in df.columns]
    if not available:
        return "数据中未找到标准水质指标。"

    latest = df[available].dropna().iloc[-1] if len(df) > 0 else None
    if latest is None:
        return "数据为空。"

    current_values = {ind: float(latest[ind]) for ind in available}
    report = generate_warnings(current_values, history_df=df)

    return ask_llm(question, context=report.summary, model=model)


def handle_report(
    question: str,
    df: Optional[pd.DataFrame],
    retriever: BaseRetriever,
    model: str,
    **_: Any,
) -> str:
    """生成综合分析报告（结构化输出）。"""
    if df is None:
        return "未找到水质监测数据，无法生成报告。"

    sections: List[str] = []

    # 1. 基础统计
    indicators = ["ph", "do", "codmn", "nh3n", "tp"]
    available = [c for c in indicators if c in df.columns]
    if available:
        desc = df[available].describe().to_string()
        sections.append(f"## 基础统计\n{desc}")

    # 2. 综合评价
    fusion_report = batch_fusion(df)
    sections.append(
        f"## 综合评价\nWQI 均值: {fusion_report.wqi_mean}, "
        f"标准差: {fusion_report.wqi_std}\n"
        f"等级分布: {json.dumps(fusion_report.level_distribution, ensure_ascii=False)}"
    )

    # 3. 趋势分析
    time_col = _find_time_col(df)
    if time_col:
        trend_report = analyze_trend(df, time_col=time_col)
        sections.append(f"## 趋势分析\n{trend_report.summary}")

    # 4. 异常检测
    anomaly_report = detect_anomalies(df)
    sections.append(
        f"## 异常检测\n异常总数: {anomaly_report.anomaly_count}, "
        f"按指标: {json.dumps(anomaly_report.summary, ensure_ascii=False)}"
    )

    # 5. 汛期分析
    if time_col:
        season_report = analyze_season(df, time_col=time_col)
        sections.append(f"## 汛期/非汛期对比\n{season_report.summary}")

    context = "\n\n".join(sections)

    system_prompt = (
        "你是水环境分析专家。根据以下多维度分析结果，生成一份结构化的水质分析报告。"
        "报告需包含：概述、各指标分析、趋势判断、异常说明、汛期影响、风险预警和建议。"
        "使用 Markdown 格式，语言专业严谨。"
    )

    return ask_llm(question, context=context, model=model, system_prompt=system_prompt)


def handle_standard(
    question: str,
    model: str,
    **_: Any,
) -> str:
    """查询水质标准。"""
    standards = get_standards()
    context = json.dumps(standards, ensure_ascii=False, indent=2, default=str)
    return ask_llm(question, context=f"地表水环境质量标准:\n{context}", model=model)


def handle_extreme(
    question: str,
    df: Optional[pd.DataFrame],
    retriever: BaseRetriever,
    model: str,
    top_k: int = 4,
    **_: Any,
) -> str:
    """极值查询。"""
    if df is not None:
        indicators = ["ph", "do", "codmn", "nh3n", "tp"]
        available = [c for c in indicators if c in df.columns]
        if available:
            extremes: Dict[str, Dict[str, float]] = {}
            for ind in available:
                series = df[ind].dropna()
                if len(series) > 0:
                    extremes[ind] = {
                        "最小值": float(series.min()),
                        "最大值": float(series.max()),
                        "均值": float(series.mean()),
                    }
            context = json.dumps(extremes, ensure_ascii=False, indent=2)
            return ask_llm(question, context=f"各指标极值统计:\n{context}", model=model)

    return handle_rag(question, retriever=retriever, model=model, top_k=top_k)


def handle_list(
    question: str,
    df: Optional[pd.DataFrame],
    retriever: BaseRetriever,
    model: str,
    top_k: int = 4,
    **_: Any,
) -> str:
    """汇总列举。"""
    if df is not None:
        indicators = ["ph", "do", "codmn", "nh3n", "tp"]
        available = [c for c in indicators if c in df.columns]
        if available:
            desc = df[available].describe().to_string()
            return ask_llm(question, context=f"数据统计汇总:\n{desc}", model=model)

    return handle_rag(question, retriever=retriever, model=model, top_k=top_k)


# ═══════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════


def _find_time_col(df: pd.DataFrame) -> Optional[str]:
    """自动检测时间列。"""
    candidates = ["date", "datetime", "time", "timestamp", "日期", "时间", "监测时间", "采样时间"]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    return None


def _find_station_col(df: pd.DataFrame) -> Optional[str]:
    """自动检测站点列。"""
    candidates = ["station", "site", "断面", "站点", "监测点", "site_name", "station_name"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ═══════════════════════════════════════════════════════════════════════════
# 意图路由
# ═══════════════════════════════════════════════════════════════════════════

HANDLERS = {
    "rag":       handle_rag,
    "evaluate":  handle_evaluate,
    "extreme":   handle_extreme,
    "list":      handle_list,
    "trend":     handle_trend,
    "anomaly":   handle_anomaly,
    "season":    handle_season,
    "spatial":   handle_spatial,
    "forecast":  handle_forecast,
    "report":    handle_report,
    "standard":  handle_standard,
}


def route_question(
    question: str,
    intent: str,
    retriever: BaseRetriever,
    df: Optional[pd.DataFrame],
    model: str,
    top_k: int = 4,
) -> str:
    """根据意图路由到对应处理函数。"""
    handler = HANDLERS.get(intent, handle_rag)
    return handler(
        question=question,
        retriever=retriever,
        df=df,
        model=model,
        top_k=top_k,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="水环境数据分析智能问答系统")
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
        default="auto",
        help="向量检索后端",
    )
    parser.add_argument("--persist-dir", default=None, help="ChromaDB 持久化目录")
    parser.add_argument(
        "--include-tabular-in-rag",
        action="store_true",
        help="将 CSV/Excel 也纳入 RAG 检索（默认仅加载文本知识）",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_file = Path(args.data_file) if args.data_file else None
    knowledge_dir = Path(args.knowledge_dir)
    if not data_dir.exists():
        logger.warning("结构化数据目录不存在: %s", data_dir)
    if not knowledge_dir.exists():
        logger.warning("知识目录不存在: %s", knowledge_dir)

    # 1. 初始化向量检索器
    retriever = create_retriever(
        backend=args.vector_backend,
        persist_dir=args.persist_dir,
    )
    chunks = load_documents(knowledge_dir, include_tabular=args.include_tabular_in_rag)
    if chunks:
        try:
            retriever.add_chunks(chunks)
            logger.info("已从 %s 加载 %d 个文档片段到检索器", knowledge_dir, retriever.count())
        except Exception as exc:
            logger.warning("向量后端初始化失败（将回退 TF-IDF）: %s", exc)
            retriever = create_retriever(backend="tfidf")
            retriever.add_chunks(chunks)
            logger.info("已回退到 TF-IDF 检索器，并加载 %d 个文档片段", retriever.count())
    else:
        logger.warning("知识目录 %s 未找到可检索文档", knowledge_dir)

    # 2. 加载结构化水质数据（用于分析模块）
    df = _load_water_data(data_dir, data_file=data_file)
    if df is not None:
        logger.info("已加载水质数据: %d 行 x %d 列", len(df), len(df.columns))
    else:
        logger.warning("未找到结构化水质数据（CSV/Excel），分析功能将受限")

    # 3. 交互循环
    print("\n" + "=" * 60)
    print("  水环境数据分析智能问答系统")
    print("  支持：水质评价 | 趋势分析 | 异常检测 | 汛期对比")
    print("        空间关联 | 预警预报 | 报告生成 | 知识检索")
    print("=" * 60)

    while True:
        try:
            question = input("\n请输入您的问题（输入 exit 退出）：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("已退出。")
            break

        intent = detect_intent(question)
        logger.info("意图识别: %s", intent)

        answer = route_question(
            question=question,
            intent=intent,
            retriever=retriever,
            df=df,
            model=args.model,
            top_k=args.top_k,
        )
        print(f"\n{answer}")


if __name__ == "__main__":
    main()
