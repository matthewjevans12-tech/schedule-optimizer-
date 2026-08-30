from __future__ import annotations

WORKSPACE_DB_VERSION = "7.1.0"

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


    # ----------------------- transaction workflow -----------------------

    def create_transaction(self, payload: Dict[str, Any]) -> str:
        """Create a school-to-school scheduling transaction."""
        stamp = datetime.now(timezone.utc)
        tx_id = payload.get("transaction_id") or f"tx_{stamp.strftime('%Y%m%d%H%M%S')}_{abs(hash(json.dumps(payload, sort_keys=True, default=str))) % 100000:05d}"
        body = dict(payload)
        body["transaction_id"] = tx_id
        body.setdefault("created_at", stamp.isoformat())
        body.setdefault("updated_at", stamp.isoformat())
        body.setdefault("history", [])
        self.put("transaction", tx_id, body)
        return tx_id

    def get_transaction(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        return self.get("transaction", transaction_id)

    def save_transaction(self, transaction_id: str, payload: Dict[str, Any]) -> None:
        body = dict(payload)
        body["transaction_id"] = transaction_id
        body["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.put("transaction", transaction_id, body)

    def list_transactions(self) -> List[Dict[str, Any]]:
        return self.list("transaction")

    def record_transaction_action(
        self,
        transaction_id: str,
        *,
        actor: str,
        action: str,
        note: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        tx = self.get_transaction(transaction_id)
        if not tx:
            return None
        history = list(tx.get("history", []))
        history.append({
            "at": datetime.now(timezone.utc).isoformat(),
            "actor": actor,
            "action": action,
            "note": note,
            "extra": extra or {},
        })
        tx["history"] = history
        self.save_transaction(transaction_id, tx)
        return tx

    def set_school_approval(
        self,
        transaction_id: str,
        school: str,
        status: str,
        note: str = "",
    ) -> Optional[Dict[str, Any]]:
        tx = self.get_transaction(transaction_id)
        if not tx:
            return None
        approvals = dict(tx.get("school_approvals", {}))
        approvals[school] = str(status).upper()
        tx["school_approvals"] = approvals
        if str(status).upper() == "REJECTED":
            tx["status"] = "REJECTED"
        tx = self._maybe_complete_transaction(tx)
        self.save_transaction(transaction_id, tx)
        self.record_transaction_action(
            transaction_id,
            actor=school,
            action=f"SCHOOL_{str(status).upper()}",
            note=note,
        )
        return self.get_transaction(transaction_id)

    def set_conference_approval(
        self,
        transaction_id: str,
        conference: str,
        status: str,
        note: str = "",
    ) -> Optional[Dict[str, Any]]:
        tx = self.get_transaction(transaction_id)
        if not tx:
            return None
        approvals = dict(tx.get("conference_approvals", {}))
        approvals[conference] = str(status).upper()
        tx["conference_approvals"] = approvals
        if str(status).upper() == "REJECTED":
            tx["status"] = "REJECTED"
        tx = self._maybe_complete_transaction(tx)
        self.save_transaction(transaction_id, tx)
        self.record_transaction_action(
            transaction_id,
            actor=conference,
            action=f"CONFERENCE_{str(status).upper()}",
            note=note,
        )
        return self.get_transaction(transaction_id)

    def _maybe_complete_transaction(self, tx: Dict[str, Any]) -> Dict[str, Any]:
        if str(tx.get("status", "")).upper() == "REJECTED":
            return tx
        school_approvals = dict(tx.get("school_approvals", {}))
        conference_approvals = dict(tx.get("conference_approvals", {}))
        schools_done = bool(school_approvals) and all(v == "ACCEPTED" for v in school_approvals.values())
        conferences_done = all(v == "ACCEPTED" for v in conference_approvals.values())
        if schools_done and conferences_done:
            tx["status"] = "COMPLETED"
            tx["completed_at"] = datetime.now(timezone.utc).isoformat()
        else:
            tx["status"] = "PENDING"
        return tx


    def set_transaction_status(
        self,
        transaction_id: str,
        status: str,
        *,
        actor: str = "System",
        note: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        tx = self.get_transaction(transaction_id)
        if not tx:
            return None
        tx["status"] = str(status).upper()
        self.save_transaction(transaction_id, tx)
        self.record_transaction_action(
            transaction_id,
            actor=actor,
            action=f"STATUS_{str(status).upper()}",
            note=note,
            extra=extra,
        )
        return self.get_transaction(transaction_id)

    def request_transaction_change(
        self,
        transaction_id: str,
        *,
        school: str,
        game_id: str,
        requested_week: int,
        note: str = "",
    ) -> Optional[Dict[str, Any]]:
        tx = self.get_transaction(transaction_id)
        if not tx:
            return None
        approvals = dict(tx.get("school_approvals", {}))
        approvals[school] = "CHANGES_REQUESTED"
        tx["school_approvals"] = approvals
        tx["status"] = "CHANGES_REQUESTED"
        suggestions = list(tx.get("suggestions", []))
        suggestions.append({
            "school": school,
            "game_id": game_id,
            "requested_week": int(requested_week),
            "note": note,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        tx["suggestions"] = suggestions
        self.save_transaction(transaction_id, tx)
        self.record_transaction_action(
            transaction_id,
            actor=school,
            action="SUGGEST_ALTERNATIVE",
            note=note,
            extra={"game_id": game_id, "requested_week": int(requested_week)},
        )
        return self.get_transaction(transaction_id)
