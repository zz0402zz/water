# 钱塘江流域水环境数据分析与预警平台

这是一个面向课程设计和后续论文扩展的 Python 原型系统，包含：

- 水质标准评价与 WQI 综合评价
- 趋势分析、异常识别、汛期对比、空间分析
- 预警生成
- 基于检索增强的大模型问答
- Markdown / JSON 报告输出

## 1. 环境准备

如果项目目录里已经有虚拟环境 `water/`，激活方式如下：

```bash
source /Users/zz/Applications/test/water/bin/activate
```

安装依赖：

```bash
uv pip install --python water/bin/python -r requirements.txt
```

## 2. 推荐目录组织

建议逐步把数据整理成下面的结构：

```text
data/
  knowledge/    # 标准、报告、制度、案例等文本资料
  datasets/     # 水质监测表、流量表、站点表等结构化数据
```

当前 `app.py` 已支持：

- `--knowledge-dir` 显式指定知识目录
- `--data-file` 显式指定结构化水质数据文件
- 默认仅将文本资料纳入 RAG
- 通过 `--include-tabular-in-rag` 才把 CSV / Excel 纳入检索

## 3. 常用命令

启动问答与分析主程序：

```bash
water/bin/python app.py
```

启动本地网页平台：

```bash
water/bin/python web_app.py
```

启动后在浏览器访问：

```text
http://127.0.0.1:8000
```

如果你想避免首次启动下载向量模型，网页端默认使用 `tfidf` 检索；需要 ChromaDB 时可手动传入：

```bash
water/bin/python web_app.py --vector-backend chroma
```

这个网页端支持：

- 浏览最新水质数据与关键指标
- 查看趋势分析、异常检测、预警结果
- 预览自动生成的综合报告
- 在页面里直接提问，而不是只在终端交互

如果你已经把知识和结构化数据分开：

```bash
water/bin/python app.py \
  --data-dir data/datasets \
  --knowledge-dir data/knowledge \
  --data-file data/datasets/water_quality.csv
```

使用命令行分析工具：

```bash
water/bin/python -m water_quality.cli report --data data/datasets/water_quality_sample.csv
```

批量预测与报告：

```bash
water/bin/python -m water_quality.cli batch-predict --data input.csv --model models/water_quality_model.pkl
water/bin/python -m water_quality.cli report --data outputs/predictions.csv --output outputs/report.md
```

## 4. 当前已修复的关键问题

- 已将核心源码统一为 UTF-8，恢复可导入状态
- 已避免默认把整个 `data/` 目录中的表格直接送入 RAG
- 已改进结构化水质数据的自动识别逻辑
- 已将批量预测能力合并进主 CLI，减少重复入口

## 5. 后续建议

优先继续完善下面几项：

- 增加真实钱塘江流域监测数据样例
- 为知识库补充标准、政策和典型案例文本
- 增加数据字段说明和最小测试集
- 继续完善网页端交互，例如文件上传、图表筛选和账号化管理
