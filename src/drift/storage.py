import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  started_at      TEXT NOT NULL,
  model           TEXT NOT NULL,
  embedding_model TEXT NOT NULL,
  kind            TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS responses (
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id         INTEGER NOT NULL REFERENCES runs(id),
  prompt_id      TEXT NOT NULL,
  prompt_text    TEXT NOT NULL,
  response_text  TEXT NOT NULL,
  embedding_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_responses_run_prompt ON responses(run_id, prompt_id);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def insert_run(
    conn: sqlite3.Connection, *, model: str, embedding_model: str, kind: str
) -> int:
    started_at = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        "INSERT INTO runs (started_at, model, embedding_model, kind) VALUES (?, ?, ?, ?)",
        (started_at, model, embedding_model, kind),
    )
    conn.commit()
    return cur.lastrowid


def insert_response(
    conn: sqlite3.Connection,
    *,
    run_id: int,
    prompt_id: str,
    prompt_text: str,
    response_text: str,
    embedding: list[float],
) -> int:
    cur = conn.execute(
        "INSERT INTO responses (run_id, prompt_id, prompt_text, response_text, embedding_json) "
        "VALUES (?, ?, ?, ?, ?)",
        (run_id, prompt_id, prompt_text, response_text, json.dumps(embedding)),
    )
    conn.commit()
    return cur.lastrowid


def get_response(conn: sqlite3.Connection, response_id: int) -> dict:
    row = conn.execute(
        "SELECT id, run_id, prompt_id, prompt_text, response_text, embedding_json "
        "FROM responses WHERE id = ?",
        (response_id,),
    ).fetchone()
    if row is None:
        raise KeyError(response_id)
    data = dict(row)
    data["embedding"] = json.loads(data.pop("embedding_json"))
    return data
