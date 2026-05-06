import sqlite3
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from drift.config import CHAT_MODEL, EMBEDDING_MODEL, load_openai_api_key
from drift.metrics import cosine
from drift.providers import openai as openai_provider
from drift.report import Comparison, build_markdown
from drift.storage import (
    connect,
    insert_response,
    insert_run,
    latest_baseline_run,
    responses_for_run,
)


class Prompt(BaseModel):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)


def _load_prompts(prompts_path: Path) -> list[Prompt]:
    raw = yaml.safe_load(prompts_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{prompts_path} must contain a YAML list of prompts")
    prompts = [Prompt.model_validate(item) for item in raw]
    seen: set[str] = set()
    for p in prompts:
        if p.id in seen:
            raise ValueError(f"duplicate prompt id: {p.id}")
        seen.add(p.id)
    return prompts


def _capture_run(
    conn: sqlite3.Connection, *, kind: str, prompts: list[Prompt]
) -> tuple[int, int]:
    """Run the prompt suite once, store results. Returns (run_id, failure_count)."""
    run_id = insert_run(
        conn, model=CHAT_MODEL, embedding_model=EMBEDDING_MODEL, kind=kind
    )
    failures = 0
    for p in prompts:
        try:
            response = openai_provider.chat(p.text)
            embedding = openai_provider.embed(response)
        except Exception as exc:
            failures += 1
            print(f"[fail] prompt_id={p.id}: {exc}")
            continue

        insert_response(
            conn,
            run_id=run_id,
            prompt_id=p.id,
            prompt_text=p.text,
            response_text=response,
            embedding=embedding,
        )
        print(f"[ok] prompt_id={p.id} ({len(response)} chars, {len(embedding)}-dim)")
    return run_id, failures


def run_baseline(prompts_path: Path, db_path: Path) -> int:
    load_openai_api_key()
    prompts = _load_prompts(prompts_path)
    conn = connect(db_path)
    run_id, failures = _capture_run(conn, kind="baseline", prompts=prompts)
    captured = len(prompts) - failures
    print(f"Baseline complete: {captured}/{len(prompts)} prompts captured. run_id={run_id}")
    return 1 if failures else 0


def _build_comparisons(
    baseline_responses: list[dict],
    eval_responses: list[dict],
    threshold: float,
) -> list[Comparison]:
    baseline_by_id = {r["prompt_id"]: r for r in baseline_responses}
    eval_by_id = {r["prompt_id"]: r for r in eval_responses}

    comparisons: list[Comparison] = []
    for r in eval_responses:
        pid = r["prompt_id"]
        if pid in baseline_by_id:
            sim = cosine(baseline_by_id[pid]["embedding"], r["embedding"])
            comparisons.append(
                Comparison(
                    prompt_id=pid,
                    kind="compared",
                    similarity=sim,
                    baseline_response=baseline_by_id[pid]["response_text"],
                    eval_response=r["response_text"],
                    passed=sim >= threshold,
                )
            )
        else:
            comparisons.append(
                Comparison(
                    prompt_id=pid,
                    kind="new",
                    similarity=None,
                    baseline_response=None,
                    eval_response=r["response_text"],
                    passed=None,
                )
            )

    for r in baseline_responses:
        pid = r["prompt_id"]
        if pid not in eval_by_id:
            comparisons.append(
                Comparison(
                    prompt_id=pid,
                    kind="missing",
                    similarity=None,
                    baseline_response=r["response_text"],
                    eval_response=None,
                    passed=None,
                )
            )
    return comparisons


def run_eval(
    prompts_path: Path, db_path: Path, threshold: float, report_dir: Path
) -> int:
    load_openai_api_key()
    prompts = _load_prompts(prompts_path)
    conn = connect(db_path)

    baseline = latest_baseline_run(conn)
    if baseline is None:
        print("No baseline run found. Run 'drift baseline' first.")
        return 1

    eval_run_id, failures = _capture_run(conn, kind="eval", prompts=prompts)

    baseline_responses = responses_for_run(conn, baseline["id"])
    eval_responses = responses_for_run(conn, eval_run_id)
    comparisons = _build_comparisons(baseline_responses, eval_responses, threshold)

    eval_run_row = conn.execute(
        "SELECT id, started_at, model, embedding_model, kind FROM runs WHERE id = ?",
        (eval_run_id,),
    ).fetchone()

    markdown = build_markdown(
        eval_run=dict(eval_run_row),
        baseline_run=baseline,
        comparisons=comparisons,
        threshold=threshold,
    )

    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"run-{eval_run_id}.md"
    report_path.write_text(markdown, encoding="utf-8")

    failed_compared = [
        c for c in comparisons if c.kind == "compared" and not c.passed
    ]
    result = "FAIL" if failed_compared else "PASS"
    print(f"Drift report: {result} — wrote {report_path}")

    return 1 if (failed_compared or failures) else 0
