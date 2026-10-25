"""数据库连接与会话管理。

SQLite 文件：data/studymate.db（随项目走，重启不丢）。
"""
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker

from database.models import Base, User

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "studymate.db"

engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={
        "check_same_thread": False,  # 多线程（Streamlit / 后台解析线程）访问 SQLite
        "timeout": 30,  # 并发写等待锁的秒数（配合 PRAGMA busy_timeout）
    },
)


@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):
    """每个新连接打开时执行 PRAGMA：
    - WAL：写不阻塞读，后台解析写库时页面读取不卡顿
    - foreign_keys=ON：SQLite 默认关闭外键，开启后约束才真正生效
    - busy_timeout：拿不到锁时等待而非立刻报 database is locked
    """
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA busy_timeout=30000")
    cursor.close()

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db() -> None:
    """建表 + 预置默认用户（幂等，可重复执行）。"""
    PROJECT_ROOT.joinpath("data").mkdir(exist_ok=True)
    Base.metadata.create_all(engine)
    _migrate()
    with SessionLocal() as db:
        if db.get(User, 1) is None:
            db.add(User(id=1, name="StudyMate 用户"))
            db.commit()


def _migrate() -> None:
    """轻量列迁移：给已存在的表补新增列（create_all 不会改旧表）。"""
    with engine.connect() as conn:
        cols = {
            row[1]
            for row in conn.execute(text("PRAGMA table_info(knowledge_documents)"))
        }
        if "chunks" not in cols:
            conn.execute(
                text("ALTER TABLE knowledge_documents ADD COLUMN chunks JSON")
            )
        conn.commit()


@contextmanager
def session_scope():
    """请求级会话：正常结束自动 commit，异常自动回滚并关闭。"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
