"""数据访问层：计划/任务/记录/会话/知识库文档的增删改查与统计。

app.py（Streamlit 界面）、agent/tools/agent_tools.py（Agent 工具）、
rag/rag_service.py 与 rag/backfill.py（知识库）都调用这里，
保证界面与 Agent 读写的是同一份 SQLite。
"""
# —— 学习笔记：函数菜谱地图 ——
# 菜谱A 转换器：对象进 → 抠字段贴标签 → 时间转字符串 → 字典出
#        精讲：_plan_dict    同款：_task_dict / _record_dict / _doc_dict
# 菜谱B 增  加：造草稿 → add → flush 拿号 → 记流水 → 交复印件
#        精讲：create_plan   同款：create_task / create_document
# 菜谱C 查：query → order_by → all/first/get → 翻译
#        精讲：list_plans / get_plan
#        收编：latest_plan / list_records / list_tasks / get_task / list_documents / get_document
# 菜谱D 删：get 问在不在 → 留念想 → delete → 报真假
#        精讲：delete_plan   同款：delete_task / delete_record（不记流水）/ delete_document（还路径）
# 菜谱E 特殊：各有各的剧情
#        update_task（部分更新+状态机）/ record_chat_turn（首轮自动建会话）
#        tasks_of_today（或条件）/ get_statistics（聚合统计）/ delete_chat_session（先批量删消息）
from datetime import date, datetime

from sqlalchemy import distinct, func
from sqlalchemy.orm import Session

from database.models import (
    ChatMessage,
    ChatSession,
    KnowledgeDocument,
    LearningRecord,
    StudyPlan,
    StudyTask,
)


def _plan_dict(p: StudyPlan) -> dict:
    return {
        "id": p.id,
        "topic": p.topic,
        "days": p.days,
        "stages": p.stages or [],
        "status": p.status,
        "created_at": p.created_at.isoformat(timespec="seconds"),
    }


def _task_dict(t: StudyTask) -> dict:
    return {
        "id": t.id,
        "title": t.title,
        "due_date": t.due_date or "",
        "status": t.status,
        "created_at": t.created_at.isoformat(timespec="seconds"),
        "completed_at": t.completed_at.isoformat(timespec="seconds") if t.completed_at else "",
    }


def _record_dict(r: LearningRecord) -> dict:
    return {
        "id": r.id,
        "time": r.created_at.isoformat(timespec="seconds"),
        "kind": r.kind,
        "entry": r.entry,
    }


# ---------- 学习记录 ----------


def add_record(db: Session, kind: str, entry: str) -> None:
    db.add(LearningRecord(kind=kind, entry=entry))


def list_records(db: Session, limit: int = 10) -> list[dict]:
    limit = min(max(limit, 1), 100)
    rows = (
        db.query(LearningRecord)
        .order_by(LearningRecord.created_at.desc(), LearningRecord.id.desc())
        .limit(limit)
        .all()
    )
    return [_record_dict(r) for r in rows]


def delete_record(db: Session, record_id: int) -> bool:
    """删除一条学习记录（最近动态）。"""
    r = db.get(LearningRecord, record_id)
    if r is None:
        return False
    db.delete(r)
    return True


# ---------- 学习计划 ----------


def create_plan(db: Session, topic: str, days: int, stages: list[dict]) -> dict:
    plan = StudyPlan(topic=topic, days=days, stages=stages)
    db.add(plan)
    db.flush()  # flush 以拿到自增 id
    add_record(db, "plan_created", f"创建计划 #{plan.id}：{topic}（{days}天）")
    return _plan_dict(plan)


def list_plans(db: Session) -> list[dict]:
    rows = db.query(StudyPlan).order_by(StudyPlan.created_at.desc()).all()
    return [_plan_dict(p) for p in rows]


def get_plan(db: Session, plan_id: int) -> dict | None:
    p = db.get(StudyPlan, plan_id)
    return _plan_dict(p) if p else None


def latest_plan(db: Session) -> dict | None:
    p = db.query(StudyPlan).order_by(StudyPlan.created_at.desc()).first()
    return _plan_dict(p) if p else None


def delete_plan(db: Session, plan_id: int) -> bool:
    """删除学习计划，并记一条学习记录。"""
    p = db.get(StudyPlan, plan_id)
    if p is None:
        return False
    topic, pid = p.topic, p.id
    db.delete(p)
    add_record(db, "plan_deleted", f"删除计划 #{pid}：{topic}")
    return True


# ---------- 学习任务 ----------


def _validate_due_date(value: str | None) -> str:
    """空日期表示不设截止时间，其余必须为真实的 YYYY-MM-DD 日期。"""
    if value is None or value == "":
        return ""
    try:
        if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
            raise ValueError
    except ValueError:
        raise ValueError("截止日期必须为有效的 YYYY-MM-DD 日期，或留空") from None
    return value


def create_task(db: Session, title: str, due_date: str | None = "") -> dict:
    due_date = _validate_due_date(due_date)
    task = StudyTask(title=title, due_date=due_date)
    db.add(task)
    db.flush()  # flush 以拿到自增 id
    add_record(db, "task_created", f"创建任务 #{task.id}：{title}")
    return _task_dict(task)


def list_tasks(db: Session, status: str | None = None) -> list[dict]:
    q = db.query(StudyTask)
    if status in ("pending", "completed"):
        q = q.filter(StudyTask.status == status)
    rows = q.order_by(StudyTask.created_at.desc()).all()
    return [_task_dict(t) for t in rows]


def get_task(db: Session, task_id: int) -> dict | None:
    t = db.get(StudyTask, task_id)
    return _task_dict(t) if t else None


def update_task(db: Session, task_id: int, fields: dict) -> dict | None:
    """部分更新；状态切到 completed 时补 completed_at 并记一条学习记录。"""
    t = db.get(StudyTask, task_id)
    if t is None:
        return None

    # 在修改任何字段前校验，避免非法日期导致部分更新。
    due_date = _validate_due_date(fields["due_date"]) if "due_date" in fields else None
    new_status = fields.get("status")
    if "title" in fields and fields["title"]:
        t.title = fields["title"]
    if "due_date" in fields:
        t.due_date = due_date

    if new_status in ("pending", "completed") and new_status != t.status:
        t.status = new_status
        if new_status == "completed":
            t.completed_at = datetime.now()
            add_record(db, "task_completed", f"完成任务 #{t.id}：{t.title}")
        else:
            t.completed_at = None
    db.flush()
    return _task_dict(t)


def delete_task(db: Session, task_id: int) -> bool:
    t = db.get(StudyTask, task_id)
    if t is None:
        return False
    title, old_id = t.title, t.id
    db.delete(t)
    add_record(db, "task_deleted", f"删除任务 #{old_id}：{title}")
    return True


def tasks_of_today(db: Session) -> list[dict]:
    """今日任务：截止日期=今天，或创建日期=今天。"""
    today = date.today().isoformat()
    rows = (
        db.query(StudyTask)
        .filter(
            (StudyTask.due_date == today)
            | (func.date(StudyTask.created_at) == today)
        )
        .order_by(StudyTask.created_at.desc())
        .all()
    )
    return [_task_dict(t) for t in rows]


# ---------- 统计 ----------


def get_statistics(db: Session) -> dict:
    study_days = (
        db.query(func.count(distinct(func.date(LearningRecord.created_at)))).scalar() or 0
    )
    tasks_total = db.query(func.count(StudyTask.id)).scalar() or 0
    tasks_completed = (
        db.query(func.count(StudyTask.id)).filter(StudyTask.status == "completed").scalar()
        or 0
    )
    plans_total = db.query(func.count(StudyPlan.id)).scalar() or 0
    return {
        "study_days": int(study_days),
        "tasks_total": int(tasks_total),
        "tasks_completed": int(tasks_completed),
        "completion_rate": (
            f"{tasks_completed / tasks_total * 100:.0f}%" if tasks_total else "0%"
        ),
        "plans_total": int(plans_total),
    }


# ---------- 会话历史 ----------


def record_chat_turn(db: Session, session_id: int | None, user_text: str, reply_text: str) -> int:
    """保存一轮对话：会话不存在则创建（标题取首条消息），返回会话 id。"""
    session = db.get(ChatSession, session_id) if session_id else None
    if session is None:
        title = user_text.strip()[:30] + ("…" if len(user_text.strip()) > 30 else "")
        session = ChatSession(title=title)
        db.add(session)
        db.flush()
    else:
        session.updated_at = datetime.now()
        if not session.title:
            session.title = user_text.strip()[:30]
    db.add(ChatMessage(session_id=session.id, role="user", content=user_text))
    db.add(ChatMessage(session_id=session.id, role="assistant", content=reply_text))
    db.flush()
    return session.id


def list_chat_sessions(db: Session, limit: int = 20) -> list[dict]:
    rows = (
        db.query(ChatSession)
        .order_by(ChatSession.updated_at.desc(), ChatSession.id.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": s.id,
            "title": s.title or "(未命名会话)",
            "updated_at": s.updated_at.isoformat(timespec="seconds"),
        }
        for s in rows
    ]


def get_chat_messages(db: Session, session_id: int) -> list[dict] | None:
    """会话消息列表；会话不存在返回 None。"""
    if db.get(ChatSession, session_id) is None:
        return None
    rows = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.id.asc())
        .all()
    )
    return [
        {
            "id": m.id,
            "role": m.role,
            "content": m.content,
            "created_at": m.created_at.isoformat(timespec="seconds"),
        }
        for m in rows
    ]


def delete_chat_session(db: Session, session_id: int) -> bool:
    """删除会话及其全部消息。"""
    session = db.get(ChatSession, session_id)
    if session is None:
        return False
    db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
    db.delete(session)
    return True


# ---------- 知识库文档 ----------


def _doc_dict(d: KnowledgeDocument, include_chunks: bool = False) -> dict:
    base = {
        "id": d.id,
        "filename": d.filename,
        "size": d.size,
        "status": d.status,
        "chunk_count": d.chunk_count,
        "error": d.error,
        "created_at": d.created_at.isoformat(timespec="seconds"),
    }
    if include_chunks:
        base["chunks"] = d.chunks or []
    return base


def create_document(db: Session, filename: str, filepath: str, size: int) -> dict:
    doc = KnowledgeDocument(filename=filename, filepath=filepath, size=size, status="processing")
    db.add(doc)
    db.flush()
    return _doc_dict(doc)


def list_documents(db: Session) -> list[dict]:
    """文档列表（不含 chunks，避免响应过大）。"""
    rows = db.query(KnowledgeDocument).order_by(KnowledgeDocument.created_at.desc()).all()
    return [_doc_dict(d) for d in rows]


def get_document(db: Session, doc_id: int, include_chunks: bool = False) -> dict | None:
    d = db.get(KnowledgeDocument, doc_id)
    return _doc_dict(d, include_chunks) if d else None


def update_document(
    db: Session,
    doc_id: int,
    status: str | None = None,
    chunk_count: int | None = None,
    chunks: list | None = None,
    error: str | None = None,
) -> None:
    """更新文档的处理状态；只覆盖传了值的字段（None 表示不动）。"""
    d = db.get(KnowledgeDocument, doc_id)
    if d is None:
        return
    if status is not None:
        d.status = status
    if chunk_count is not None:
        d.chunk_count = chunk_count
    if chunks is not None:
        d.chunks = chunks
    if error is not None:
        d.error = error


def delete_document(db: Session, doc_id: int) -> str | None:
    """删除文档记录，返回文件路径（供调用方删文件）；不存在返回 None。"""
    d = db.get(KnowledgeDocument, doc_id)
    if d is None:
        return None
    filepath = d.filepath
    db.delete(d)
    return filepath
