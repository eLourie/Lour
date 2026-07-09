"""
app/infra/db/base.py

SQLAlchemy declarative base with consistent constraint-naming conventions.
All ORM models import Base from here — never create a second one.
"""

from __future__ import annotations

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

# Alembic autogenerate and PostgreSQL both need explicit constraint names.
# This convention ensures all FK / unique / check constraints are named
# deterministically, so migrations are reproducible.
NAMING_CONVENTION: dict[str, str] = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """
    Project-wide declarative base.

    All ORM model classes inherit from this. The shared MetaData
    with naming conventions is attached so Alembic autogenerate
    produces stable, conflict-free constraint names.
    """

    metadata = MetaData(naming_convention=NAMING_CONVENTION)
