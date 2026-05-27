"""
database.py
─────────────────────────────────────────────────────────────────────────────
Database engine, session factory, declarative base, and FastAPI dependency.

Every module that needs a database session imports get_db() from here.
Nothing else in the codebase creates its own engine or session directly.

Environment variable required:
    DATABASE_URL — full PostgreSQL connection string.
    Format: postgresql://user:password@host:port/dbname

Railway example:
    DATABASE_URL=postgresql://postgres:abc123@containers-us-west-1.railway.app:6543/railway

Local development example:
    DATABASE_URL=postgresql://postgres:password@localhost:5432/reviewsnipper
─────────────────────────────────────────────────────────────────────────────
"""

import os
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase
from dotenv import load_dotenv

# Load .env file when running locally.
# On Railway this is a no-op — env vars are injected by the platform.
load_dotenv()


# ── Read database URL from environment ───────────────────────────────────────

DATABASE_URL: str = os.environ.get("DATABASE_URL", "")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL environment variable is not set.\n"
        "Set it in your .env file (local) or Railway dashboard (production).\n"
        "Format: postgresql://user:password@host:port/dbname"
    )

# SQLAlchemy requires 'postgresql://' not 'postgres://'.
# Railway sometimes provides the older 'postgres://' prefix — fix it silently.
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)


# ── Create the database engine ────────────────────────────────────────────────
#
# pool_pre_ping=True
#   Before handing a connection from the pool to a request, SQLAlchemy sends
#   a lightweight SELECT 1 to verify the connection is still alive.
#   This prevents "server closed the connection unexpectedly" errors after
#   Railway or the database server restarts.
#
# pool_size=5, max_overflow=10
#   5 persistent connections in the pool.
#   Up to 10 additional connections created on burst demand.
#   Appropriate for a Railway starter instance with 512MB–1GB RAM.
#
# connect_args={"options": "-c timezone=UTC"}
#   Force the PostgreSQL session timezone to UTC.
#   This ensures all timestamp fields are stored and returned in UTC
#   regardless of the server's local timezone setting.

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    connect_args={"options": "-c timezone=UTC"},
    echo=False,   # Set to True temporarily to log all SQL — useful for debugging
)


# ── Session factory ───────────────────────────────────────────────────────────
#
# autocommit=False
#   We manage transactions explicitly.  Changes are not written to the
#   database until db.commit() is called.  If an error occurs before commit,
#   the transaction is automatically rolled back when the session closes.
#
# autoflush=False
#   SQLAlchemy will not automatically flush pending changes to the database
#   before each query.  We flush manually when needed.  This prevents
#   accidental partial writes during complex multi-step operations.

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)


# ── Declarative base ──────────────────────────────────────────────────────────
#
# All SQLAlchemy model classes inherit from Base.
# Alembic reads Base.metadata to detect schema changes and generate migrations.

class Base(DeclarativeBase):
    pass


# ── FastAPI database dependency ───────────────────────────────────────────────
#
# Usage in any FastAPI route:
#
#     from app.database import get_db
#     from sqlalchemy.orm import Session
#
#     @router.get("/example")
#     def example_route(db: Session = Depends(get_db)):
#         results = db.query(SomeModel).all()
#         return results
#
# FastAPI calls next(get_db()) to open a session at the start of each request
# and automatically calls the finally block to close it when the response
# is sent — even if an exception occurred during the request.
#
# This guarantees:
#   ✓ One session per request — no session sharing between requests
#   ✓ Session always closed — no connection pool exhaustion
#   ✓ Failed transactions rolled back — no partial data corruption

def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a database session for a single request.

    Opens a session, yields it to the route handler, then closes it
    in the finally block regardless of success or failure.
    """
    db = SessionLocal()
    try:
        yield db
    except Exception:
        # Roll back any uncommitted changes if an error occurred
        db.rollback()
        raise
    finally:
        db.close()
