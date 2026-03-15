from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .model import WaterQualityModel
from .pipeline import load_measurements, predict_dataframe
from .standards import evaluate_level, explain_indicator, get_standards
from .trend import analyze_trend
from .anomaly import detect_anomalies
from .fusion import compute_wqi, batch_fusion
from .warning import generate_warnings, ALERT_LABELS
from .report import generate_report, save_report, FullReport


DEFAULT_FEATURES = ["ph", "do", "codmn", "nh3n", "tp"]


def _resolve_time_col(df: pd.DataFrame, requested: str | None) -> str | None:
    if requested and requested in df.columns:
        return requested

    for column in ["date", "datetime", "time", "timestamp", "日期", "时间", "监测时间", "采样时间"]:
        if column in df.columns:
            return column

    return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Water quality prediction + standards tool")
    sub = parser.add_subparsers(dest="command", required=True)

    init_cmd = sub.add_parser("init-sample", help="Create a sample CSV dataset")
    init_cmd.add_argument("--out", default="data/datasets/water_quality_sample.csv")
    init_cmd.add_argument("--rows", type=int, default=200)

    train_cmd = sub.add_parser("train", help="Train a water quality model")
    train_cmd.add_argument("--data", default="data/datasets/water_quality_sample.csv")
    train_cmd.add_argument("--model-out", default="models/water_quality_model.pkl")
    train_cmd.add_argument("--label-col", default="quality_level")
    train_cmd.add_argument("--feature-cols", nargs="*", default=DEFAULT_FEATURES)

    pred_cmd = sub.add_parser("predict", help="Predict water quality class")
    pred_cmd.add_argument("--model", default="models/water_quality_model.pkl")
    pred_cmd.add_argument("--ph", type=float, required=True)
    pred_cmd.add_argument("--do", type=float, required=True)
    pred_cmd.add_argument("--codmn", type=float, required=True)
    pred_cmd.add_argument("--nh3n", type=float, required=True)
    pred_cmd.add_argument("--tp", type=float, required=True)

    std_cmd = sub.add_parser("standard", help="Query standard thresholds")
    std_cmd.add_argument("--indicator", default=None, help="ph/do/codmn/nh3n/tp")

    eval_cmd = sub.add_parser("evaluate", help="Evaluate class by standards")
    eval_cmd.add_argument("--ph", type=float, required=True)
    eval_cmd.add_argument("--do", type=float, required=True)
    eval_cmd.add_argument("--codmn", type=float, required=True)
    eval_cmd.add_argument("--nh3n", type=float, required=True)
    eval_cmd.add_argument("--tp", type=float, required=True)

    batch_cmd = sub.add_parser("batch-predict", help="Predict water quality for a CSV dataset")
    batch_cmd.add_argument("--data", required=True, help="Input CSV path")
    batch_cmd.add_argument("--model", required=True, help="Model path")
    batch_cmd.add_argument("--out", default="outputs/predictions.csv", help="Output CSV path")

    trend_cmd = sub.add_parser("trend", help="Analyze historical trends")
    trend_cmd.add_argument("--data", default="data/datasets/water_quality_sample.csv")
    trend_cmd.add_argument("--time-col", default="date")

    anomaly_cmd = sub.add_parser("anomaly", help="Detect anomalies")
    anomaly_cmd.add_argument("--data", default="data/datasets/water_quality_sample.csv")
    anomaly_cmd.add_argument("--level", default="III", help="Exceedance comparison level")

    fusion_cmd = sub.add_parser("fusion", help="Multi-dimensional fusion evaluation")
    fusion_cmd.add_argument("--data", default="data/datasets/water_quality_sample.csv")

    warn_cmd = sub.add_parser("warn", help="Generate water quality warnings")
    warn_cmd.add_argument("--data", default="data/datasets/water_quality_sample.csv")
    warn_cmd.add_argument("--ph", type=float, required=True)
    warn_cmd.add_argument("--do", type=float, required=True)
    warn_cmd.add_argument("--codmn", type=float, required=True)
    warn_cmd.add_argument("--nh3n", type=float, required=True)
    warn_cmd.add_argument("--tp", type=float, required=True)

    report_cmd = sub.add_parser("report", help="Generate comprehensive report")
    report_cmd.add_argument("--data", default="data/datasets/water_quality_sample.csv")
    report_cmd.add_argument("--time-col", default="date")
    report_cmd.add_argument("--output", default="reports/water_quality_report.md")
    report_cmd.add_argument("--use-llm-commentary", dest="use_llm_commentary", action="store_true", help="Use LLM for AI commentary")
    report_cmd.add_argument("--use-langchain", dest="use_llm_commentary", action="store_true", help=argparse.SUPPRESS)
    report_cmd.add_argument("--model", default="deepseek-chat")

    return parser


def _generate_sample_csv(out_path: Path, rows: int = 200) -> None:
    import random

    out_path.parent.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, float | str]] = []
    classes = ["I", "II", "III", "IV", "V"]
    start_date = datetime(2024, 1, 1)

    for idx in range(rows):
        level = random.choice(classes)
        if level == "I":
            rec: Dict[str, float | str] = {"ph": random.uniform(6.8, 8.2), "do": random.uniform(7.5, 10.0), "codmn": random.uniform(1.0, 2.0), "nh3n": random.uniform(0.01, 0.15), "tp": random.uniform(0.005, 0.02)}
        elif level == "II":
            rec = {"ph": random.uniform(6.6, 8.6), "do": random.uniform(6.0, 8.0), "codmn": random.uniform(2.0, 4.0), "nh3n": random.uniform(0.15, 0.5), "tp": random.uniform(0.02, 0.1)}
        elif level == "III":
            rec = {"ph": random.uniform(6.3, 8.8), "do": random.uniform(5.0, 6.5), "codmn": random.uniform(4.0, 6.0), "nh3n": random.uniform(0.5, 1.0), "tp": random.uniform(0.1, 0.2)}
        elif level == "IV":
            rec = {"ph": random.uniform(6.1, 8.9), "do": random.uniform(3.0, 5.1), "codmn": random.uniform(6.0, 10.0), "nh3n": random.uniform(1.0, 1.5), "tp": random.uniform(0.2, 0.3)}
        else:
            rec = {"ph": random.uniform(5.8, 9.2), "do": random.uniform(1.0, 3.2), "codmn": random.uniform(10.0, 18.0), "nh3n": random.uniform(1.5, 3.0), "tp": random.uniform(0.3, 0.6)}
        rec["date"] = (start_date + timedelta(days=idx)).strftime("%Y-%m-%d")
        rec["quality_level"] = level
        records.append(rec)

    pd.DataFrame(records).to_csv(out_path, index=False, encoding="utf-8")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "init-sample":
        out = Path(args.out)
        _generate_sample_csv(out, rows=args.rows)
        print(f"Sample dataset created: {out}")
        return

    if args.command == "train":
        data_path = Path(args.data)
        model_path = Path(args.model_out)
        wq_model = WaterQualityModel()
        result = wq_model.train(
            data_path=data_path,
            label_col=args.label_col,
            feature_cols=args.feature_cols,
        )
        wq_model.save(model_path)
        print(f"Model saved to: {model_path}")
        print(f"Accuracy: {result.accuracy:.4f}")
        print(result.report)
        return

    if args.command == "predict":
        model = WaterQualityModel.load(Path(args.model))
        features = {
            "ph": args.ph,
            "do": args.do,
            "codmn": args.codmn,
            "nh3n": args.nh3n,
            "tp": args.tp,
        }
        model_result = model.predict(features)
        std_result = evaluate_level(features)
        print("Model result:", model_result)
        print("Standards evaluation:", std_result)
        return

    if args.command == "standard":
        if args.indicator:
            print(explain_indicator(args.indicator))
        else:
            print(get_standards())
        return

    if args.command == "evaluate":
        features = {
            "ph": args.ph,
            "do": args.do,
            "codmn": args.codmn,
            "nh3n": args.nh3n,
            "tp": args.tp,
        }
        print(evaluate_level(features))
        return

    if args.command == "batch-predict":
        df = load_measurements(Path(args.data))
        result = predict_dataframe(df, model_path=Path(args.model))
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_path, index=False, encoding="utf-8")
        print(f"Predictions saved: {out_path}")
        return

    if args.command == "trend":
        df = pd.read_csv(args.data)
        time_col = _resolve_time_col(df, args.time_col)
        if time_col is None:
            raise ValueError("数据中未找到时间列，请通过 --time-col 显式指定。")
        report = analyze_trend(df, time_col=time_col)
        print("=== 趋势分析结果 ===")
        for ind, t in report.trends.items():
            print(f"  {ind}: {t.trend} (斜率={t.slope:.6f}, p={t.p_value:.4f}, 变化率={t.change_rate:.1f}%)")
        if report.seasonals:
            print("\n=== 季节性分析 ===")
            for ind, s in report.seasonals.items():
                print(f"  {ind}: 强度={s.seasonal_strength:.2f}, 高峰={s.peak_month}月, 低谷={s.trough_month}月")
        print(f"\n摘要: {report.summary}")
        return

    if args.command == "anomaly":
        df = pd.read_csv(args.data)
        report = detect_anomalies(df, exceedance_level=args.level)
        print(f"=== 异常检测结果 ===")
        print(f"总记录: {report.total_records}, 异常数: {report.anomaly_count}")
        for ind, count in report.summary.items():
            print(f"  {ind}: {count} 处异常")
        if report.anomalies:
            print("\n前 10 条异常:")
            for a in report.anomalies[:10]:
                print(f"  [{a.severity}] 行 {a.index}, {a.indicator}={a.value:.4f} ({a.method}: {a.reason})")
        return

    if args.command == "fusion":
        df = pd.read_csv(args.data)
        report = batch_fusion(df)
        print(f"=== 数据融合评价 ===")
        print(f"WQI 均值: {report.wqi_mean}, 标准差: {report.wqi_std}")
        if report.level_distribution:
            print("等级分布:")
            for level, count in sorted(report.level_distribution.items()):
                print(f"  {level} 类: {count}")
        if report.pca_explained_variance:
            print("PCA 方差解释:", [f"{v:.2%}" for v in report.pca_explained_variance])
        return

    if args.command == "warn":
        features = {
            "ph": args.ph,
            "do": args.do,
            "codmn": args.codmn,
            "nh3n": args.nh3n,
            "tp": args.tp,
        }
        history_df = pd.read_csv(args.data)
        report = generate_warnings(features, history_df=history_df)
        print(f"=== 水质预警 ===")
        print(report.summary)
        return

    if args.command == "report":
        df = pd.read_csv(args.data)
        output_path = Path(args.output)
        time_col = _resolve_time_col(df, args.time_col)
        report = generate_report(
            df,
            time_col=time_col,
            use_llm_commentary=args.use_llm_commentary,
            model_name=args.model,
        )
        assert isinstance(report, FullReport), "generate_report should return FullReport when out_path is None"
        save_report(report, output_path)
        print(f"报告已生成: {output_path}")
        return


if __name__ == "__main__":
    main()
