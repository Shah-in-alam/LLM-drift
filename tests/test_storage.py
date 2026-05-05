from drift.storage import connect, get_response, insert_response, insert_run


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
