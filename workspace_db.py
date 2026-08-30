from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, text


class WorkspaceDB:
    """Small persistence layer.

    Production: set DATABASE_URL to Postgres.
    Pilot fallback: local SQLite. Streamlit Community Cloud local disk may be
    replaced on redeploy, so SQLite should not be treated as durable production storage.
    """

    def __init__(self, database_url: Optional[str] = None):
        url = database_url or os.getenv("DATABASE_URL") or "sqlite:///pilot_workspace.db"
        if url.startswith("postgres://"):
            url = "postgresql+psycopg://" + url[len("postgres://"):]
        elif url.startswith("postgresql://"):
            url = "postgresql+psycopg://" + url[len("postgresql://"):]
        self.url = url
        self.engine = create_engine(url, future=True, pool_pre_ping=True)
        self._init()

    @property
    def durable(self) -> bool:
        return not self.url.startswith("sqlite:")

    def _init(self):
        ddl = [
            """
            CREATE TABLE IF NOT EXISTS kv_store (
                namespace VARCHAR(80) NOT NULL,
                item_key VARCHAR(240) NOT NULL,
                payload TEXT NOT NULL,
                updated_at VARCHAR(40) NOT NULL,
                PRIMARY KEY(namespace, item_key)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at VARCHAR(40) NOT NULL,
                season INTEGER,
                team VARCHAR(240),
                game_id VARCHAR(240),
                reason VARCHAR(120) NOT NULL,
                notes TEXT,
                payload TEXT
            )
            """ if self.url.startswith("sqlite:") else """
            CREATE TABLE IF NOT EXISTS feedback (
                id BIGSERIAL PRIMARY KEY,
                created_at VARCHAR(40) NOT NULL,
                season INTEGER,
                team VARCHAR(240),
                game_id VARCHAR(240),
                reason VARCHAR(120) NOT NULL,
                notes TEXT,
                payload TEXT
            )
            """,
        ]
        with self.engine.begin() as conn:
            for stmt in ddl:
                conn.execute(text(stmt))

    def put(self, namespace: str, key: str, payload: Dict[str, Any]):
        now = datetime.now(timezone.utc).isoformat()
        body = json.dumps(payload, default=str)
        with self.engine.begin() as conn:
            conn.execute(
                text("DELETE FROM kv_store WHERE namespace=:n AND item_key=:k"),
                {"n": namespace, "k": key},
            )
            conn.execute(
                text("INSERT INTO kv_store(namespace,item_key,payload,updated_at) VALUES(:n,:k,:p,:u)"),
                {"n": namespace, "k": key, "p": body, "u": now},
            )

    def get(self, namespace: str, key: str) -> Optional[Dict[str, Any]]:
        with self.engine.begin() as conn:
            row = conn.execute(
                text("SELECT payload FROM kv_store WHERE namespace=:n AND item_key=:k"),
                {"n": namespace, "k": key},
            ).first()
        return json.loads(row[0]) if row else None

    def list(self, namespace: str) -> List[Dict[str, Any]]:
        with self.engine.begin() as conn:
            rows = conn.execute(
                text("SELECT item_key,payload,updated_at FROM kv_store WHERE namespace=:n ORDER BY updated_at DESC"),
                {"n": namespace},
            ).all()
        return [
            {"key": r[0], "payload": json.loads(r[1]), "updated_at": r[2]}
            for r in rows
        ]

    def delete(self, namespace: str, key: str):
        with self.engine.begin() as conn:
            conn.execute(
                text("DELETE FROM kv_store WHERE namespace=:n AND item_key=:k"),
                {"n": namespace, "k": key},
            )

    def add_feedback(
        self,
        *,
        season: Optional[int],
        team: str,
        game_id: str,
        reason: str,
        notes: str,
        payload: Optional[Dict[str, Any]] = None,
    ):
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO feedback(created_at,season,team,game_id,reason,notes,payload) "
                    "VALUES(:c,:s,:t,:g,:r,:n,:p)"
                ),
                {
                    "c": datetime.now(timezone.utc).isoformat(),
                    "s": season,
                    "t": team,
                    "g": game_id,
                    "r": reason,
                    "n": notes,
                    "p": json.dumps(payload or {}, default=str),
                },
            )
