"""
ChromaDB 向量库（langchain-chroma 封装）：持久化到 data/chroma/。

- 入库：add_document() —— 切块向量化后写入（幂等，同文档先清后写）
- 删除：delete_document() —— 文档删除时同步清掉它的向量
- 检索：search() —— 查询向量化后返回最相关的 top-k 条
- 切块：split_to_chunks() —— langchain 分片器按页切块，保留页码元数据
"""
import threading

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.config import Settings
from chromadb.telemetry.product import ProductTelemetryClient
from overrides import override

from model.factory import get_embed_model
from utils.config_handler import chroma_conf
from utils.logger_handler import logger
from utils.path_tool import get_abs_path

BATCH = 200  # 单次写入批大小

# 补丁类
# ChromaDB 默认会向自家服务器发送匿名使用统计（"遥测"），底层用 posthog 这个统计库。
class _NullTelemetry(ProductTelemetryClient):
    """遥测空实现：chroma 内置的 posthog 客户端版本不兼容会刷红错，用它替换掉。"""

    @override
    def capture(self, event) -> None:
        pass


class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=get_embed_model(),
            persist_directory=get_abs_path(chroma_conf["persist_directory"]),
            collection_metadata={"hnsw:space": "cosine"},
            client_settings=Settings(
                anonymized_telemetry=False,
                chroma_product_telemetry_impl=f"{__name__}._NullTelemetry",
            ),
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

    def split_to_chunks(self, documents: list[Document]) -> list[dict]:
        """
        PDF 页 Document 列表 → 切块列表：[{page, index, text}]
        page 为 1 基页码（与数据库 chunks 字段格式一致），跳过空块。
        """
        if not documents:
            return []

        pieces = self.splitter.split_documents(documents)

        out: list[dict] = []
        for d in pieces:
            text = d.page_content.strip()
            if not text:
                continue
            out.append(
                {
                    "page": int(d.metadata.get("page", 0)) + 1,
                    "index": len(out),
                    "text": text,
                }
            )
        return out

    def add_document(self, doc_id: int, filename: str, chunks: list[dict]) -> int:
        """把一份文档的切块向量化并写入（幂等：同文档先删后写；空 chunks 等价于清空该文档向量）。"""
        self.delete_document(doc_id)
        if not chunks:
            return 0

        documents: list[Document] = []
        ids: list[str] = []
        for i, c in enumerate(chunks):
            documents.append(
                Document(
                    page_content=c["text"],
                    metadata={
                        "doc_id": doc_id,
                        "filename": filename,
                        "page": int(c.get("page", 0)),
                        "index": int(c.get("index", i)),
                    },
                )
            )
            ids.append(f"doc{doc_id}_c{c.get('index', i)}")

        total = 0
        for start in range(0, len(documents), BATCH):
            self.vector_store.add_documents(
                documents[start : start + BATCH], ids=ids[start : start + BATCH]
            )
            total += min(BATCH, len(documents) - start)

        logger.info(f"[vector_store]文档{doc_id}（{filename}）已向量化入库{total}块")
        return total

    def delete_document(self, doc_id: int) -> None:
        """删除该文档的全部向量（文档删除时调用）。"""
        self.vector_store.delete(where={"doc_id": doc_id})

    def has_document(self, doc_id: int, chunks: list[dict]) -> bool:
        """检查该文档的全部分块 ID，避免部分写入被误判为完成。"""
        expected = {f"doc{doc_id}_c{c.get('index', i)}" for i, c in enumerate(chunks)}
        res = self.vector_store.get(where={"doc_id": doc_id}, include=[])
        return bool(expected) and set(res.get("ids", [])) == expected

    def search(self, query: str, k: int = 4, min_score: float = 0.0) -> list[dict]:
        """语义检索 top-k：返回 [{text, filename, page, score}]，score < min_score 的片段被过滤。"""
        if not query.strip():
            return []

        # Chroma 返回 cosine 距离；转换为相似度，并消除浮点误差造成的越界。
        scored = [
            (doc, max(0.0, min(1.0, 1.0 - distance)))
            for doc, distance in self.vector_store.similarity_search_with_score(query, k=k)
        ]
        return [
            {
                "text": d.page_content,
                "filename": d.metadata.get("filename", ""),
                "page": d.metadata.get("page", 0),
                "score": float(score),
            }
            for d, score in scored
            if score >= min_score
        ]


_store = None
_store_lock = threading.Lock()


def get_vector_store() -> VectorStoreService:
    """向量库服务单例（延迟构造 + 双重检查锁，保证多线程下只构造一次）。"""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = VectorStoreService()
    return _store


if __name__ == '__main__':
    store = get_vector_store()
    for r in store.search("数据结构", k=2):
        print(f"《{r['filename']}》第{r['page']}页：{r['text'][:60]}...")
        print("-" * 20)
