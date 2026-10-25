"""
资料文件加载：PDF → langchain Document 列表（保留页码元数据，供 RAG 标注"资料来源"页码）
"""
import pymupdf
from langchain_core.documents import Document
from utils.logger_handler import logger


def pdf_loader(filepath: str) -> list[Document]:
    """
    用 PyMuPDF 按页读取 PDF，每页转为一个 Document
    metadata["page"] 为 0 基页码（入库时转成 1 基展示）
    :param filepath: PDF 文件路径
    :return: Document 列表
    """
    documents: list[Document] = []

    try:
        with pymupdf.open(filepath) as doc:   #PDF 文档对象本身可迭代——每迭代一次出一页（page）
            for i, page in enumerate(doc):   #enumerate(可迭代对象) 是 Python 的内置函数，用于在遍历可迭代对象时同时获得索引和元素值
                text = page.get_text() or ""
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"page": i, "source": filepath},
                    )
                )
    except Exception as e:
        logger.error(f"[pdf_loader]读取PDF失败：{filepath}，原因：{str(e)}")
        raise e

    return documents


if __name__ == '__main__':
    from utils.path_tool import get_abs_path

    docs = pdf_loader(get_abs_path("data/数据结构示例资料.pdf"))
    print(f"共 {len(docs)} 页，第一页前80字：")
    print(docs[0].page_content[:80])
