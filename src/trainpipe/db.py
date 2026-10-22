"""Engine y sesiones. Postgres; no se usa SQLite (habria que ir en modo WAL
y dos escritores concurrentes darian `database is locked`)."""

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from trainpipe.config import get_settings

_engine = None
_SessionLocal = None


def get_engine():
    global _engine, _SessionLocal
    if _engine is None:
        _engine = create_engine(
            get_settings().database_url,
            pool_pre_ping=True,  # evita conexiones muertas tras un restart de db
            future=True,
        )
        _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def get_session() -> Iterator[Session]:
    """Dependencia de FastAPI. Cierra siempre."""
    get_engine()
    session = _SessionLocal()
    try:
        yield session
    finally:
        session.close()


def session_factory() -> Session:
    """Sesion para uso fuera del ciclo de request (worker, CLI)."""
    get_engine()
    return _SessionLocal()