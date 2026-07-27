import sqlite3
from pathlib import Path

import pytest

from powerhouse_ai.schemas import DocumentAdd, Domain
from powerhouse_ai.storage import MemoryStore


def test_project_memory_and_document_retrieval(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "test.db")
    project = store.create_project("Bacteria")
    store.add_memory(
        project.id,
        "The simulation uses logistic growth and hours as the time unit.",
    )
    memories = store.search_memory(project.id, "Which growth model uses hours?")
    assert memories
    assert "logistic growth" in memories[0]
    assert store.latest_memories(project.id) == [
        "The simulation uses logistic growth and hours as the time unit."
    ]

    store.add_document(
        DocumentAdd(
            project_id=project.id,
            domain=Domain.BIOLOGY,
            title="ATP synthase notes",
            content="ATP synthase uses a proton gradient and a rotational mechanism.",
            source="local notes",
            trust_level="high",
        )
    )
    evidence = store.search_documents(
        "How does the proton gradient drive ATP synthase?",
        [Domain.BIOLOGY],
        project_id=project.id,
    )
    assert evidence
    assert evidence[0].trust_level == "high"


def test_memory_is_isolated_between_projects(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "test.db")
    one = store.create_project("One")
    two = store.create_project("Two")
    store.add_memory(one.id, "secret zebra decision")
    assert store.search_memory(two.id, "zebra") == []


def test_memory_search_rejects_non_positive_limit(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "test.db")
    project = store.create_project("Limits")
    with pytest.raises(ValueError, match="at least 1"):
        store.search_memory(project.id, "anything", limit=-1)


def test_existing_database_gets_evidence_link_migration(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as database:
        database.execute("CREATE TABLE task_steps (id TEXT PRIMARY KEY)")

    MemoryStore(path)

    with sqlite3.connect(path) as database:
        columns = {
            row[1] for row in database.execute("PRAGMA table_info(task_steps)").fetchall()
        }
    assert "evidence_ids_json" in columns
