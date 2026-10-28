"""
知识库检索服务：用户提问，向量检索资料片段，整理成 Agent 可用的结构化结果
（chunks + 来源页码 + 作答约束说明，由 Agent 基于片段生成带"资料来源"标注的回答）
"""
from database import repo
from database.db import session_scope
from rag.vector_store import get_vector_store
from utils.config_handler import chroma_conf


class KnowledgeSearchService(object):
    def __init__(self):
        self.vector_store = get_vector_store()

    def search(self, query: str) -> dict:
        """
        语义检索已上传资料（RAG 检索环节）。

        返回 top-4 相关片段（含文件名+页码），低于相关度阈值的片段被过滤。
        库为空或无达标片段时如实返回 found=False；片段中无答案时由提示词约束 LLM 不得编造。
        """
        if not query.strip():
            return {"ok": False, "error": "query 不能为空"}

        results = self.vector_store.search(
            query, k=chroma_conf["k"], min_score=chroma_conf.get("min_score", 0.3)
        )
        if not results:
            with session_scope() as db:
                docs = repo.list_documents(db)
            note = (
                "知识库为空，尚未上传任何资料。"
                if not docs
                else "没有找到足够相关的资料片段，或资料尚未完成索引。"
            )
            return {"ok": True, "found": False, "chunks": [], "sources": [], "note": note}

        chunks = [
            {
                "filename": r["filename"],
                "page": r["page"],
                "text": r["text"][:600],
            }
            for r in results
        ]
        # 去重来源（文件+页）
        sources = []
        seen = set()
        for r in results:
            key = (r["filename"], r["page"])
            if key not in seen:
                seen.add(key)
                sources.append({"filename": r["filename"], "page": r["page"]})

        return {
            "ok": True,
            "found": True,
            "chunks": chunks,
            "sources": sources,
            "note": (
                f"检索到 {len(chunks)} 个相关片段。回答必须且只能基于这些片段；"
                "若片段中没有所需信息，明确说明'资料中未涉及'，不得将通用知识说成来自资料。"
            ),
        }


if __name__ == '__main__':
    service = KnowledgeSearchService()
    print(service.search("数据结构"))
