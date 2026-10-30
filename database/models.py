"""数据表模型（SQLAlchemy 2.x 声明式）。

共 7 张表：
    User / StudyPlan / StudyTask / LearningRecord / KnowledgeDocument
    ChatSession / ChatMessage（会话历史）
当前为单用户模式（启动时预置一条 User），暂不做登录注册。
"""
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """SQLAlchemy 2.x 声明式基类。"""


class User(Base):
    """用户表（预留，单用户模式）。"""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(50), default="StudyMate 用户")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class StudyPlan(Base):
    """学习计划表：stages 以 JSON 存储（阶段列表）。"""

    __tablename__ = "study_plans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    topic: Mapped[str] = mapped_column(String(200))
    days: Mapped[int] = mapped_column(Integer)
    stages: Mapped[list] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(20), default="active")  # active/done
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class StudyTask(Base):
    """学习任务表。"""

    __tablename__ = "study_tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    due_date: Mapped[str] = mapped_column(String(20), default="")  # YYYY-MM-DD，可为空
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending/completed
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class LearningRecord(Base):
    """学习记录表：任务和计划事件；保留旧时长字段以兼容已有数据库。"""

    __tablename__ = "learning_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String(30))  # plan_created/task_created/task_completed/task_deleted
    entry: Mapped[str] = mapped_column(Text, default="")
    seconds: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class ChatSession(Base):
    """聊天会话表（会话历史）：title 取首条用户消息。"""

    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(60), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class ChatMessage(Base):
    """聊天消息表：会话内的逐条 user/assistant 消息。"""

    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(20))  # user/assistant
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class KnowledgeDocument(Base):
    """知识库文档表：status 是处理状态，chunks 存解析切分后的文本块。"""

    __tablename__ = "knowledge_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    filepath: Mapped[str] = mapped_column(String(500))
    size: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending/processing/ready/failed
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    chunks: Mapped[list] = mapped_column(JSON, default=list)  # [{page, index, text}]
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
