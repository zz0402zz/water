"""向量数据库 RAG 模块。

支持两种后端：
  1. ChromaDB — 生产级向量数据库，持久化存储
  2. TF-IDF  — 纯内存回退方案，无需额外依赖

自动检测 chromadb 是否可用，不可用时回退至 TF-IDF。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """文档片段。"""

    source: str
    text: str
    metadata: Optional[dict[str, Any]] = None


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------


class BaseRetriever(ABC):
    """检索器统一接口。"""

    @abstractmethod
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """向索引中添加文档片段。"""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 4) -> List[Tuple[Chunk, float]]:
        """检索 top_k 个最相关文档片段。"""

    @abstractmethod
    def count(self) -> int:
        """返回索引中的文档片段总数。"""


# ---------------------------------------------------------------------------
# TF-IDF 检索器（回退方案）
# ---------------------------------------------------------------------------


class TfidfRetriever(BaseRetriever):
    """基于 TF-IDF + 余弦相似度的检索器。"""

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.chunks: List[Chunk] = []
        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5))
        self.matrix = None

    def add_chunks(self, chunks: List[Chunk]) -> None:
        self.chunks.extend(chunks)
        if self.chunks:
            self.matrix = self.vectorizer.fit_transform([c.text for c in self.chunks])

    def retrieve(self, query: str, top_k: int = 4) -> List[Tuple[Chunk, float]]:
        if not self.chunks or self.matrix is None:
            return []
        from sklearn.metrics.pairwise import cosine_similarity
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix).flatten()
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.chunks[i], float(s)) for i, s in ranked if s > 0]

    def count(self) -> int:
        return len(self.chunks)


# ---------------------------------------------------------------------------
# ChromaDB 检索器
# ---------------------------------------------------------------------------


class ChromaRetriever(BaseRetriever):
    """基于 ChromaDB 的向量数据库检索器。

    支持持久化存储，适合大规模知识库和历史数据。
    """

    def __init__(
        self,
        collection_name: str = "water_quality_kb",
        persist_dir: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> None:
        import chromadb  # type: ignore[import-unresolved]

        if persist_dir:
            self.client = chromadb.PersistentClient(path=persist_dir)
        else:
            self.client = chromadb.Client()

        # 使用 chromadb 内置 embedding 或自定义
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._chunk_map: dict[str, Chunk] = {}

    def add_chunks(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[dict[str, Any]] = []

        existing_count = self.collection.count()

        for i, chunk in enumerate(chunks):
            doc_id = f"doc_{existing_count + i}"
            ids.append(doc_id)
            documents.append(chunk.text)
            meta = {"source": chunk.source}
            if chunk.metadata:
                meta.update(chunk.metadata)
            metadatas.append(meta)
            self._chunk_map[doc_id] = chunk

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,  # type: ignore[assignment]
        )
        logger.info("ChromaDB: 已添加 %d 个片段，总计 %d", len(chunks), self.collection.count())

    def retrieve(self, query: str, top_k: int = 4) -> List[Tuple[Chunk, float]]:
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count()),
        )

        output: List[Tuple[Chunk, float]] = []
        if results and results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            distances = results["distances"][0] if results["distances"] else [0.0] * len(ids)
            documents = results["documents"][0] if results["documents"] else [""] * len(ids)
            metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(ids)

            for doc_id, dist, doc_text, meta in zip(ids, distances, documents, metadatas):
                # ChromaDB 返回的是距离（越小越好），转为相似度
                similarity = max(0.0, 1.0 - dist)
                chunk = self._chunk_map.get(doc_id)
                if chunk is None:
                    chunk = Chunk(
                        source=str(meta.get("source", "unknown")),
                        text=doc_text,
                        metadata=meta,  # type: ignore[assignment]
                    )
                output.append((chunk, similarity))

        return output

    def count(self) -> int:
        return self.collection.count()


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------


def create_retriever(
    backend: str = "auto",
    persist_dir: Optional[str] = None,
    collection_name: str = "water_quality_kb",
) -> BaseRetriever:
    """创建检索器。

    Parameters
    ----------
    backend : str
        "chroma", "tfidf", 或 "auto"（自动检测）。
    persist_dir : str, optional
        ChromaDB 持久化目录。
    collection_name : str
        ChromaDB collection 名称。

    Returns
    -------
    BaseRetriever
    """
    if backend == "auto":
        try:
            import chromadb  # type: ignore[import-unresolved]  # noqa: F401
            backend = "chroma"
            logger.info("检测到 chromadb，使用向量数据库后端")
        except ImportError:
            backend = "tfidf"
            logger.info("chromadb 未安装，回退到 TF-IDF 后端")

    if backend == "chroma":
        return ChromaRetriever(
            collection_name=collection_name,
            persist_dir=persist_dir,
        )
    else:
        return TfidfRetriever()


# ---------------------------------------------------------------------------
# 文档加载工具
# ---------------------------------------------------------------------------

TEXT_EXTENSIONS = {".txt", ".md", ".markdown"}
TABULAR_EXTENSIONS = {".xls", ".xlsx", ".csv"}


def split_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    """将长文本拆分为重叠片段。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须为正整数")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap 必须 >= 0 且 < chunk_size")

    text = " ".join(text.split())
    if len(text) <= chunk_size:
        return [text]

    pieces: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        pieces.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return pieces


def load_documents(data_dir: Path, include_tabular: bool = False) -> List[Chunk]:
    """从目录递归加载文档并切分成 Chunk。

    默认仅加载适合语义检索的文本资料，表格文件需显式开启。
    """
    import pandas as pd

    chunks: List[Chunk] = []
    if not data_dir.exists():
        return chunks

    supported_extensions = set(TEXT_EXTENSIONS)
    if include_tabular:
        supported_extensions.update(TABULAR_EXTENSIONS)

    for file_path in sorted(data_dir.rglob("*")):
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        if suffix not in supported_extensions:
            continue

        try:
            if suffix in TEXT_EXTENSIONS:
                text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
            elif suffix in {".xls", ".xlsx"}:
                engine = "xlrd" if suffix == ".xls" else "openpyxl"
                df = pd.read_excel(file_path, engine=engine)
                text = df.to_string()
            elif suffix == ".csv":
                last_error: Optional[Exception] = None
                for encoding in ("utf-8", "utf-8-sig", "gb18030"):
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        text = df.to_string()
                        break
                    except Exception as exc:
                        last_error = exc
                else:
                    assert last_error is not None
                    raise last_error
            else:
                continue
        except Exception as exc:
            logger.warning("跳过文件 %s: %s", file_path, exc)
            continue

        if not text:
            continue
        for piece in split_text(text):
            chunks.append(Chunk(
                source=str(file_path),
                text=piece,
                metadata={"file_type": suffix},
            ))
    return chunks
