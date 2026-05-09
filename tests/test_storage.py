import sqlite3

from drift.storage import (
    connect,
    get_response,
    insert_response,
    insert_run,
    latest_baseline_run,
    recent_eval_runs,
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
    assert "samples" in cols
    assert "temperature" in cols

    resp_cols = {row["name"] for row in conn.execute("PRAGMA table_info(responses)").fetchall()}
    assert "sample_idx" in resp_cols

    baseline = latest_baseline_run(conn)
    assert baseline is not None
    assert baseline["provider"] == "openai"
    assert baseline["samples"] == 1
    assert baseline["temperature"] == 0.0


def test_insert_run_persists_samples_and_temperature(tmp_path):
    conn = connect(tmp_path / "test.db")
    run_id = insert_run(
        conn,
        model="m",
        embedding_model="e",
        kind="baseline",
        provider="openai",
        samples=3,
        temperature=0.7,
    )
    row = conn.execute("SELECT samples, temperature FROM runs WHERE id = ?", (run_id,)).fetchone()
    assert row["samples"] == 3
    assert row["temperature"] == 0.7


def test_recent_eval_runs_returns_newest_first_limited(tmp_path):
    conn = connect(tmp_path / "test.db")
    insert_run(conn, model="m", embedding_model="e", kind="baseline", provider="openai")
    eval_ids = [
        insert_run(conn, model="m", embedding_model="e", kind="eval", provider="openai")
        for _ in range(4)
    ]
    insert_run(conn, model="m", embedding_model="e", kind="baseline", provider="openai")

    rows = recent_eval_runs(conn, limit=2)
    assert [r["id"] for r in rows] == [eval_ids[-1], eval_ids[-2]]
    assert all(r["kind"] == "eval" for r in rows)


def test_recent_eval_runs_empty_when_no_eval_runs(tmp_path):
    conn = connect(tmp_path / "test.db")
    insert_run(conn, model="m", embedding_model="e", kind="baseline", provider="openai")
    assert recent_eval_runs(conn, limit=5) == []


def test_insert_response_persists_sample_idx(tmp_path):
    conn = connect(tmp_path / "test.db")
    run_id = insert_run(
        conn,
        model="m",
        embedding_model="e",
        kind="baseline",
        provider="openai",
        samples=2,
        temperature=0.5,
    )
    for i in range(2):
        insert_response(
            conn,
            run_id=run_id,
            prompt_id="p",
            prompt_text="q",
            response_text=f"r{i}",
            embedding=[float(i), 0.0],
            sample_idx=i,
        )
    rows = conn.execute(
        "SELECT sample_idx, response_text FROM responses WHERE run_id = ? ORDER BY sample_idx",
        (run_id,),
    ).fetchall()
    assert [(r["sample_idx"], r["response_text"]) for r in rows] == [(0, "r0"), (1, "r1")]
