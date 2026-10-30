"""
Agent 工具集：7 个工具（langchain @tool 声明，schema 由类型注解 + Field 描述自动生成）。

处理函数负责真正干活，数据经 database/repo.py 写入 SQLite，返回统一为 JSON 字符串。

铁律：工具如实返回数据——查不到就说查不到，绝不编造（RAG 检索同理）；
任何异常都转为错误信息回传给模型，绝不向上抛。
"""
import json
from typing import Annotated

from pydantic import Field

from langchain_core.tools import tool

from database import repo
from database.db import session_scope
from rag.rag_service import KnowledgeSearchService


def _dumps(result: dict) -> str:
    return json.dumps(result, ensure_ascii=False)


def _safe(func, **kwargs) -> str:
    """执行工具函数并转 JSON 字符串；异常也回传给模型而不是中断服务。"""
    try:
        return _dumps(func(**kwargs))
    except TypeError as e:
        return _dumps({"ok": False, "error": f"参数不匹配: {e}"})
    except Exception as e:
        return _dumps({"ok": False, "error": f"工具执行失败: {e}"})


@tool(description="创建并保存一个学习计划。当用户要求制定/生成学习计划时调用。")
def create_study_plan(
    topic: Annotated[str, Field(description="学习主题，如：数据结构")],
    days: Annotated[int, Field(description="计划总天数，如 30")],
    stages: Annotated[list[dict] | None, Field(description="可选。阶段划分；不传则自动均分（最多4个阶段，天数不足4时阶段数=天数）")] = None,
) -> str:
    """创建学习计划。stages 可选，不传则自动均分为最多 4 个阶段（阶段天数之和恒等于 days）。"""

    def _create(topic: str, days: int, stages: list[dict] | None) -> dict:
        if not topic.strip():
            return {"ok": False, "error": "topic 不能为空"}
        if days < 1 or days > 365:
            return {"ok": False, "error": "天数需在 1-365 之间"}

        if not stages:
            # 均分：阶段数 n=min(4, days)，前 extra 个阶段各多 1 天，天数之和恒等于 days
            names = ["基础入门", "核心进阶", "强化练习", "复习冲刺"]
            n = min(4, days)
            base, extra = divmod(days, n)
            built, start = [], 1
            for i in range(n):
                length = base + (1 if i < extra else 0)
                built.append(
                    {
                        "name": names[i],
                        "days": length,
                        "range": f"Day {start}-{start + length - 1}",
                        "focus": f"{topic}{names[i]}",
                    }
                )
                start += length
            stages = built
        else:
            stages = [
                {
                    "name": str(s.get("name", f"阶段{i + 1}")),
                    "days": int(s.get("days", 0) or 0),
                    "range": str(s.get("range", "")),
                    "focus": str(s.get("focus", "")),
                }
                for i, s in enumerate(stages)
            ]
            # 校验：阶段天数之和必须等于总天数，且每阶段至少 1 天
            total = sum(s["days"] for s in stages)
            if not stages:
                return {"ok": False, "error": "stages 不能为空列表"}
            if any(s["days"] < 1 for s in stages):
                return {"ok": False, "error": "每个阶段天数必须≥1"}
            if total != days:
                return {
                    "ok": False,
                    "error": f"阶段天数之和（{total}）必须等于计划总天数（{days}）",
                }

        with session_scope() as db:
            plan = repo.create_plan(db, topic, days, stages)
        return {"ok": True, "plan_id": plan["id"], "plan": plan}

    return _safe(_create, topic=topic, days=days, stages=stages)


@tool(description="查询已保存的学习计划。用户问'我的计划'时调用。")
def get_study_plan(
    plan_id: Annotated[int | None, Field(description="计划id；不传则返回最新的")] = None,
) -> str:
    """查询学习计划。不传 plan_id 时返回最新的一个。"""

    def _get(plan_id: int | None) -> dict:
        with session_scope() as db:
            if plan_id is None:
                plan = repo.latest_plan(db)
            else:
                plan = repo.get_plan(db, plan_id)
        if plan is None:
            if plan_id is None:
                return {"ok": False, "error": "暂无学习计划，请先创建"}
            return {"ok": False, "error": f"未找到 id={plan_id} 的计划"}
        return {"ok": True, "plan": plan}

    return _safe(_get, plan_id=plan_id)


@tool(description="创建一个学习任务。当用户要求添加/新建任务时调用。")
def create_task(
    title: Annotated[str, Field(description="任务标题")],
    due_date: Annotated[str | None, Field(description="可选，截止日期 YYYY-MM-DD")] = None,
) -> str:
    """创建学习任务。"""

    def _create(title: str, due_date: str | None) -> dict:
        if not title.strip():
            return {"ok": False, "error": "title 不能为空"}
        with session_scope() as db:
            task = repo.create_task(db, title, due_date or "")
        return {"ok": True, "task": task}

    return _safe(_create, title=title, due_date=due_date)


@tool(description=(
    "把指定任务标记为已完成。当用户说'完成/做完了某个任务'时调用。"
    "若对话中不知道任务id，先调用 get_learning_records 查找"
    "（记录格式含 #任务id），再用查到的 id 调用本工具。"
))
def complete_task(
    task_id: Annotated[int, Field(description="任务id")],
) -> str:
    """把任务标记为已完成。"""

    def _complete(task_id: int) -> dict:
        with session_scope() as db:
            task = repo.get_task(db, task_id)
            if task is None:
                return {"ok": False, "error": f"未找到 id={task_id} 的任务"}
            if task["status"] == "completed":
                return {"ok": True, "task": task, "note": "该任务已是完成状态"}
            task = repo.update_task(db, task_id, {"status": "completed"})
        return {"ok": True, "task": task}

    return _safe(_complete, task_id=task_id)


@tool(description=(
    "查询最近的学习记录。用户问'我最近学了什么/学习记录'时调用；"
    "记录条目含 #任务id/#计划id，也用于查找任务id。"
))
def get_learning_records(
    limit: Annotated[int, Field(description="返回条数，默认10")] = 10,
) -> str:
    """查询最近的计划和任务学习记录。"""

    def _records(limit: int) -> dict:
        with session_scope() as db:
            records = repo.list_records(db, int(limit or 10))
        return {"ok": True, "total": len(records), "records": records}

    return _safe(_records, limit=limit)


@tool(description="获取学习统计（学习天数/任务数/完成率）。用户问学习数据/统计时调用。")
def get_statistics() -> str:
    """统计学习数据：学习天数、任务数、完成率、计划数。"""

    def _stats() -> dict:
        with session_scope() as db:
            return {"ok": True, **repo.get_statistics(db)}

    return _safe(_stats)


@tool(description=(
    "在用户上传的 PDF 资料库中检索相关内容。当问题涉及'我上传的/我的教材/资料里怎么说'时调用。"
    "必须基于检索结果回答，检索不到就如实说明。"
))
def knowledge_search(
    query: Annotated[str, Field(description="检索关键词或问题")],
) -> str:
    """语义检索已上传资料（RAG 检索环节），返回含页码的片段。"""

    def _search(query: str) -> dict:
        service = KnowledgeSearchService()
        return service.search(query)

    return _safe(_search, query=query)


TOOLS = [
    create_study_plan,
    get_study_plan,
    create_task,
    complete_task,
    get_learning_records,
    get_statistics,
    knowledge_search,
]


if __name__ == '__main__':
    print(get_statistics().invoke({}))
    print(knowledge_search.invoke({"query": "数据结构"})[:300])
