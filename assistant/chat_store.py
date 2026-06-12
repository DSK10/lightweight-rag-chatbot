import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ChatStore:
    def __init__(self, db_path: str = "data/chats.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL DEFAULT 'New chat',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    backend TEXT,
                    model TEXT,
                    awaiting_clarification INTEGER DEFAULT 0,
                    original_query TEXT DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    used_rag INTEGER DEFAULT 0,
                    sources_json TEXT DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                """
            )

    def create_conversation(self, backend: str = "", model: str = "", title: str = "New chat") -> int:
        now = _now()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO conversations (title, created_at, updated_at, backend, model)
                VALUES (?, ?, ?, ?, ?)
                """,
                (title, now, now, backend, model),
            )
            return int(cur.lastrowid)

    def list_conversations(self, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, title, created_at, updated_at, backend, model
                FROM conversations
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_conversation(self, conversation_id: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
        return dict(row) if row else None

    def update_conversation(
        self,
        conversation_id: int,
        *,
        title: str | None = None,
        awaiting_clarification: bool | None = None,
        original_query: str | None = None,
    ) -> None:
        now = _now()
        fields = ["updated_at = ?"]
        values: list = [now]

        if title is not None:
            fields.append("title = ?")
            values.append(title)
        if awaiting_clarification is not None:
            fields.append("awaiting_clarification = ?")
            values.append(1 if awaiting_clarification else 0)
        if original_query is not None:
            fields.append("original_query = ?")
            values.append(original_query)

        values.append(conversation_id)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE conversations SET {', '.join(fields)} WHERE id = ?",
                values,
            )

    def delete_conversation(self, conversation_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))

    def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        *,
        used_rag: bool = False,
        sources: list | None = None,
    ) -> int:
        now = _now()
        sources_json = json.dumps(sources or [])
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO messages (conversation_id, role, content, used_rag, sources_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (conversation_id, role, content, 1 if used_rag else 0, sources_json, now),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )
            return int(cur.lastrowid)

    def get_messages(self, conversation_id: int) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content, used_rag, sources_json, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id ASC
                """,
                (conversation_id,),
            ).fetchall()

        messages = []
        for row in rows:
            sources = json.loads(row["sources_json"] or "[]")
            messages.append(
                {
                    "role": row["role"],
                    "content": row["content"],
                    "used_rag": bool(row["used_rag"]),
                    "sources": sources,
                }
            )
        return messages

    def message_count(self, conversation_id: int) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        return int(row["n"]) if row else 0
