"""启动回填：把"已解析但还没向量化"的文档补进 ChromaDB。

只处理 status=ready 且向量库中没有的文档；幂等，每次启动都可安全执行。
"""
from database import repo
from database.db import session_scope
from rag.vector_store import get_vector_store
from utils.logger_handler import logger


def has_vectors(doc_id: int) -> bool:
    """该文档是否已在向量库中。"""
    return get_vector_store().has_document(doc_id)


def backfill_index() -> None:
    try:
        with session_scope() as db:
            ready_ids = [
                d["id"] for d in repo.list_documents(db) if d["status"] == "ready"
            ]
            pending = []
            for doc_id in ready_ids:
                if has_vectors(doc_id):
                    continue
                doc = repo.get_document(db, doc_id, include_chunks=True)
                if doc and doc.get("chunks"):
                    pending.append((doc["id"], doc["filename"], doc["chunks"]))

        # 先收集再向量化：embedding 较慢，不占着数据库事务做
        store = get_vector_store()
        for doc_id, filename, chunks in pending:
            store.add_document(doc_id, filename, chunks)
            logger.info(f"[backfill] 文档 {doc_id}（{filename}）已补向量化 {len(chunks)} 块")
    except Exception as e:  # 回填失败不阻断服务启动
        logger.error(f"[backfill] 回填失败（不影响启动）: {e}")
