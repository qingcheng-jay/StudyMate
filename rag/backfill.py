"""启动回填：把"已解析但还没向量化"的文档补进 ChromaDB。

只处理 status=ready 的文档；缺失或不完整的向量会在启动时重新写入。
"""
from database import repo
from database.db import session_scope
from rag.vector_store import get_vector_store
from utils.logger_handler import logger


def backfill_index() -> None:
    try:
        with session_scope() as db:
            ready_ids = [d["id"] for d in repo.list_documents(db) if d["status"] == "ready"]
            docs = [repo.get_document(db, doc_id, include_chunks=True) for doc_id in ready_ids]

        # 先收集再向量化：embedding 较慢，不占着数据库事务做
        store = get_vector_store()
        for doc in docs:
            if not doc or not doc.get("chunks"):
                logger.warning("[backfill] 已就绪文档缺少文本分块，无法回填")
                continue
            doc_id, filename, chunks = doc["id"], doc["filename"], doc["chunks"]
            if store.has_document(doc_id, chunks):
                continue
            try:
                store.add_document(doc_id, filename, chunks)
                logger.info(f"[backfill] 文档 {doc_id}（{filename}）已补向量化 {len(chunks)} 块")
            except Exception as e:
                logger.error(f"[backfill] 文档 {doc_id} 回填失败: {e}")
    except Exception as e:  # 回填失败不阻断服务启动
        logger.error(f"[backfill] 回填失败（不影响启动）: {e}")
