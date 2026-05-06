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
    first = insert_run(conn, model="m", embedding_model="e", kind="baseline")
    insert_run(conn, model="m", embedding_model="e", kind="eval")
    second = insert_run(conn, model="m", embedding_model="e", kind="baseline")
    latest = latest_baseline_run(conn)
    assert latest is not None
    assert latest["id"] == second
    assert latest["id"] != first


def test_responses_for_run_deserializes_embeddings(tmp_path):
    conn = connect(tmp_path / "test.db")
    run_id = insert_run(conn, model="m", embedding_model="e", kind="baseline")
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
