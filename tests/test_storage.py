import sqlite3

from drift.storage import (
    connect,
    get_response,
    insert_response,
    insert_run,
    latest_baseline_run,
    responses_for_run,
)


def test_storage_roundtrip(tmp_path):
    db_path = tmp_path / "test.db"
    conn = connect(db_path)

    run_id = insert_run(
        conn,
        model="gpt-4o-mini",
        embedding_model="text-embedding-3-large",
        kind="baseline",
        provider="openai",
    )
    assert run_id == 1

    embedding = [0.1, 0.2, 0.3, 0.4]
    response_id = insert_response(
        conn,
        run_id=run_id,
        prompt_id="greet",
        prompt_text="Say hello.",
        response_text="Hello!",
        embedding=embedding,
    )

    row = get_response(conn, response_id)
    assert row["run_id"] == run_id
    assert row["prompt_id"] == "greet"
    assert row["prompt_text"] == "Say hello."
    assert row["response_text"] == "Hello!"
    assert row["embedding"] == embedding


def test_latest_baseline_run_empty(tmp_path):
    conn = connect(tmp_path / "test.db")
    assert latest_baseline_run(conn) is None


def test_latest_baseline_run_picks_newest(tmp_path):
    conn = connect(tmp_path / "test.db")
    first = insert_run(conn, model="m", embedding_model="e", kind="baseline", provider="openai")
    insert_run(conn, model="m", embedding_model="e", kind="eval", provider="openai")
    second = insert_run(conn, model="m", embedding_model="e", kind="baseline", provider="anthropic")
    latest = latest_baseline_run(conn)
    assert latest is not None
    assert latest["id"] == second
    assert latest["id"] != first
    assert latest["provider"] == "anthropic"


def test_responses_for_run_deserializes_embeddings(tmp_path):
    conn = connect(tmp_path / "test.db")
    run_id = insert_run(conn, model="m", embedding_model="e", kind="baseline", provider="openai")
    insert_response(
        conn,
        run_id=run_id,
        prompt_id="a",
        prompt_text="qa",
        response_text="ra",
        embedding=[1.0, 2.0],
    )
    insert_response(
        conn,
        run_id=run_id,
        prompt_id="b",
        prompt_text="qb",
        response_text="rb",
        embedding=[3.0, 4.0],
    )
    rows = responses_for_run(conn, run_id)
    assert [r["prompt_id"] for r in rows] == ["a", "b"]
    assert rows[0]["embedding"] == [1.0, 2.0]
    assert rows[1]["embedding"] == [3.0, 4.0]


def test_legacy_db_migrates_provider_column(tmp_path):
    """A v0.2-era DB (no provider column) should gain it on connect()."""
    db_path = tmp_path / "legacy.db"

    # Build a v0.2 schema by hand: runs table without `provider`.
    raw = sqlite3.connect(str(db_path))
    raw.executescript(
        """
        CREATE TABLE runs (
          id              INTEGER PRIMARY KEY AUTOINCREMENT,
          started_at      TEXT NOT NULL,
          model           TEXT NOT NULL,
          embedding_model TEXT NOT NULL,
          kind            TEXT NOT NULL
        );
        CREATE TABLE responses (
          id             INTEGER PRIMARY KEY AUTOINCREMENT,
          run_id         INTEGER NOT NULL REFERENCES runs(id),
          prompt_id      TEXT NOT NULL,
          prompt_text    TEXT NOT NULL,
          response_text  TEXT NOT NULL,
          embedding_json TEXT NOT NULL
        );
        INSERT INTO runs (started_at, model, embedding_model, kind)
        VALUES ('2026-04-01T00:00:00+00:00', 'gpt-4o-mini', 'text-embedding-3-large', 'baseline');
        """
    )
    raw.commit()
    raw.close()

    conn = connect(db_path)
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
    assert "provider" in cols

    baseline = latest_baseline_run(conn)
    assert baseline is not None
    assert baseline["provider"] == "openai"
