from __future__ import annotations

import json
import re
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from powerhouse_ai.schemas import (
    DocumentAdd,
    Domain,
    Evidence,
    Project,
    RouteDecision,
    StepResult,
    TaskPlan,
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _fts_query(text: str) -> str:
    terms = re.findall(r"[\w-]{2,}", text.lower(), flags=re.UNICODE)
    return " OR ".join(f'"{term}"' for term in terms[:24])


def _chunks(text: str, size: int = 900, overlap: int = 100) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
        start += size - overlap
    return chunks


class MemoryStore:
    """SQLite persistence with FTS5 search for zero-VRAM project and document retrieval."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connection() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system', 'tool')),
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_project_created
                    ON messages(project_id, created_at);

                CREATE TABLE IF NOT EXISTS goals (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    goal TEXT NOT NULL,
                    domains_json TEXT NOT NULL,
                    complexity REAL NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    completed_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_goals_project_created
                    ON goals(project_id, created_at);

                CREATE TABLE IF NOT EXISTS task_steps (
                    id TEXT PRIMARY KEY,
                    goal_id TEXT NOT NULL REFERENCES goals(id) ON DELETE CASCADE,
                    step_number INTEGER NOT NULL,
                    domain TEXT NOT NULL,
                    objective TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result TEXT NOT NULL DEFAULT '',
                    tool_results_json TEXT NOT NULL DEFAULT '[]',
                    evidence_ids_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    completed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    goal_id TEXT REFERENCES goals(id) ON DELETE SET NULL,
                    path TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS memory_items (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    importance REAL NOT NULL DEFAULT 0.5,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                    id UNINDEXED, project_id UNINDEXED, content, tokenize='porter unicode61'
                );

                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    project_id TEXT,
                    domain TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    trust_level TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_documents_scope_domain
                    ON documents(project_id, domain);
                CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                    id UNINDEXED, title, content, tokenize='porter unicode61'
                );
                """
            )
            task_step_columns = {
                row["name"] for row in db.execute("PRAGMA table_info(task_steps)").fetchall()
            }
            if "evidence_ids_json" not in task_step_columns:
                db.execute(
                    "ALTER TABLE task_steps "
                    "ADD COLUMN evidence_ids_json TEXT NOT NULL DEFAULT '[]'"
                )

    def create_project(self, name: str, description: str = "") -> Project:
        project_id = uuid.uuid4().hex
        timestamp = _now()
        with self.connection() as db:
            db.execute(
                """
                INSERT INTO projects(id, name, description, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (project_id, name, description, timestamp, timestamp),
            )
        return Project(
            id=project_id,
            name=name,
            description=description,
            created_at=datetime.fromisoformat(timestamp),
            updated_at=datetime.fromisoformat(timestamp),
        )

    def get_project(self, project_id: str) -> Project | None:
        with self.connection() as db:
            row = db.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        return self._project_from_row(row) if row else None

    def list_projects(self) -> list[Project]:
        with self.connection() as db:
            rows = db.execute("SELECT * FROM projects ORDER BY updated_at DESC").fetchall()
        return [self._project_from_row(row) for row in rows]

    @staticmethod
    def _project_from_row(row: sqlite3.Row) -> Project:
        return Project(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def add_message(
        self,
        project_id: str,
        role: str,
        content: str,
        metadata: dict | None = None,
    ) -> str:
        message_id = uuid.uuid4().hex
        timestamp = _now()
        with self.connection() as db:
            db.execute(
                """
                INSERT INTO messages(id, project_id, role, content, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    project_id,
                    role,
                    content,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    timestamp,
                ),
            )
            db.execute(
                "UPDATE projects SET updated_at = ? WHERE id = ?",
                (timestamp, project_id),
            )
        return message_id

    def add_turn(
        self,
        project_id: str,
        user_content: str,
        assistant_content: str,
        *,
        assistant_metadata: dict | None = None,
    ) -> tuple[str, str]:
        """Persist a completed user/assistant turn atomically."""
        user_id = uuid.uuid4().hex
        assistant_id = uuid.uuid4().hex
        timestamp = _now()
        with self.connection() as db:
            db.execute(
                """
                INSERT INTO messages(id, project_id, role, content, metadata_json, created_at)
                VALUES (?, ?, 'user', ?, '{}', ?)
                """,
                (user_id, project_id, user_content, timestamp),
            )
            db.execute(
                """
                INSERT INTO messages(id, project_id, role, content, metadata_json, created_at)
                VALUES (?, ?, 'assistant', ?, ?, ?)
                """,
                (
                    assistant_id,
                    project_id,
                    assistant_content,
                    json.dumps(assistant_metadata or {}, ensure_ascii=False),
                    timestamp,
                ),
            )
            db.execute(
                "UPDATE projects SET updated_at = ? WHERE id = ?",
                (timestamp, project_id),
            )
        return user_id, assistant_id

    def recent_messages(self, project_id: str, limit: int = 12) -> list[dict[str, str]]:
        with self.connection() as db:
            rows = db.execute(
                """
                SELECT role, content FROM messages
                WHERE project_id = ?
                ORDER BY created_at DESC, rowid DESC LIMIT ?
                """,
                (project_id, limit),
            ).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]

    def add_memory(
        self,
        project_id: str,
        content: str,
        *,
        kind: str = "project_state",
        importance: float = 0.5,
        metadata: dict | None = None,
    ) -> str:
        memory_id = uuid.uuid4().hex
        with self.connection() as db:
            db.execute(
                """
                INSERT INTO memory_items(
                    id, project_id, kind, content, importance, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    project_id,
                    kind,
                    content,
                    importance,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    _now(),
                ),
            )
            db.execute(
                "INSERT INTO memory_fts(id, project_id, content) VALUES (?, ?, ?)",
                (memory_id, project_id, content),
            )
        return memory_id

    def search_memory(self, project_id: str, query: str, limit: int = 8) -> list[str]:
        if limit < 1:
            raise ValueError("Memory search limit must be at least 1.")
        match = _fts_query(query)
        if not match:
            return []
        with self.connection() as db:
            rows = db.execute(
                """
                SELECT m.content
                FROM memory_fts
                JOIN memory_items AS m ON m.id = memory_fts.id
                WHERE memory_fts MATCH ? AND m.project_id = ?
                ORDER BY bm25(memory_fts), m.importance DESC
                LIMIT ?
                """,
                (match, project_id, limit),
            ).fetchall()
        return [row["content"] for row in rows]

    def latest_memories(self, project_id: str, limit: int = 3) -> list[str]:
        with self.connection() as db:
            rows = db.execute(
                """
                SELECT content
                FROM memory_items
                WHERE project_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (project_id, limit),
            ).fetchall()
        return [row["content"] for row in rows]

    def record_run(
        self,
        project_id: str,
        route: RouteDecision,
        plan: TaskPlan,
        results: list[StepResult],
    ) -> str:
        goal_id = uuid.uuid4().hex
        timestamp = _now()
        results_by_step = {result.step.id: result for result in results}
        with self.connection() as db:
            db.execute("UPDATE goals SET active = 0 WHERE project_id = ?", (project_id,))
            db.execute(
                """
                INSERT INTO goals(
                    id, project_id, goal, domains_json, complexity,
                    active, created_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (
                    goal_id,
                    project_id,
                    route.goal,
                    json.dumps(
                        [item.name.value for item in route.domains],
                        ensure_ascii=False,
                    ),
                    route.complexity,
                    timestamp,
                    timestamp,
                ),
            )
            for step in plan.steps:
                result = results_by_step.get(step.id)
                db.execute(
                    """
                    INSERT INTO task_steps(
                        id, goal_id, step_number, domain, objective, status,
                        result, tool_results_json, evidence_ids_json, created_at, completed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        uuid.uuid4().hex,
                        goal_id,
                        step.id,
                        step.domain.value,
                        step.goal,
                        "completed" if result else "pending",
                        result.content if result else "",
                        json.dumps(
                            result.tool_results if result else [],
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            result.evidence_ids if result else [],
                            ensure_ascii=False,
                        ),
                        timestamp,
                        timestamp if result else None,
                    ),
                )
        return goal_id

    def project_state(self, project_id: str) -> dict:
        with self.connection() as db:
            goal = db.execute(
                """
                SELECT * FROM goals
                WHERE project_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (project_id,),
            ).fetchone()
            if goal is None:
                return {"project_id": project_id, "latest_goal": None, "steps": []}
            steps = db.execute(
                """
                SELECT step_number, domain, objective, status, result,
                       tool_results_json, evidence_ids_json
                FROM task_steps
                WHERE goal_id = ?
                ORDER BY step_number
                """,
                (goal["id"],),
            ).fetchall()
        step_records = []
        for row in steps:
            record = dict(row)
            record["tool_results"] = json.loads(record.pop("tool_results_json"))
            record["evidence_ids"] = json.loads(record.pop("evidence_ids_json"))
            step_records.append(record)
        return {
            "project_id": project_id,
            "latest_goal": {
                "id": goal["id"],
                "goal": goal["goal"],
                "domains": json.loads(goal["domains_json"]),
                "complexity": goal["complexity"],
                "active": bool(goal["active"]),
                "created_at": goal["created_at"],
                "completed_at": goal["completed_at"],
            },
            "steps": step_records,
        }

    def add_document(self, document: DocumentAdd) -> list[str]:
        ids: list[str] = []
        with self.connection() as db:
            for index, content in enumerate(_chunks(document.content)):
                document_id = uuid.uuid4().hex
                ids.append(document_id)
                db.execute(
                    """
                    INSERT INTO documents(
                        id, project_id, domain, title, content, source,
                        trust_level, chunk_index, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document_id,
                        document.project_id,
                        document.domain.value,
                        document.title,
                        content,
                        document.source,
                        document.trust_level,
                        index,
                        _now(),
                    ),
                )
                db.execute(
                    "INSERT INTO documents_fts(id, title, content) VALUES (?, ?, ?)",
                    (document_id, document.title, content),
                )
        return ids

    def search_documents(
        self,
        query: str,
        domains: list[Domain],
        *,
        project_id: str | None,
        limit: int = 8,
    ) -> list[Evidence]:
        match = _fts_query(query)
        if not match:
            return []
        domain_values = [domain.value for domain in domains] or [Domain.GENERAL.value]
        placeholders = ",".join("?" for _ in domain_values)
        scope_clause = "(d.project_id IS NULL OR d.project_id = ?)"
        sql = f"""
            SELECT d.*, bm25(documents_fts) AS rank
            FROM documents_fts
            JOIN documents AS d ON d.id = documents_fts.id
            WHERE documents_fts MATCH ?
              AND {scope_clause}
              AND d.domain IN ({placeholders})
            ORDER BY rank
            LIMIT ?
        """
        parameters = [match, project_id, *domain_values, limit]
        with self.connection() as db:
            rows = db.execute(sql, parameters).fetchall()
        return [
            Evidence(
                id=row["id"],
                title=row["title"],
                content=row["content"],
                domain=Domain(row["domain"]),
                source=row["source"],
                trust_level=row["trust_level"],
                score=1.0 / (1.0 + abs(float(row["rank"]))),
            )
            for row in rows
        ]
