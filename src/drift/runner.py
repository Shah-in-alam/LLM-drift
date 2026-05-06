import sqlite3
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from drift.metrics import cosine
from drift.providers import ChatProvider, get_embedder, get_provider
from drift.providers.openai_embed import EMBEDDING_MODEL
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
    conn: sqlite3.Connection,
    *,
    kind: str,
    prompts: list[Prompt],
    provider: ChatProvider,
) -> tuple[int, int]:
    """Run the prompt suite once, store results. Returns (run_id, failure_count)."""
    embed = get_embedder()
    run_id = insert_run(
        conn,
        model=provider.chat_model,
        embedding_model=EMBEDDING_MODEL,
        kind=kind,
        provider=provider.name,
    )
    failures = 0
    for p in prompts:
        try:
            response = provider.chat(p.text)
            embedding = embed(response)
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


def run_baseline(prompts_path: Path, db_path: Path, provider_name: str) -> int:
    provider = get_provider(provider_name)
    prompts = _load_prompts(prompts_path)
    conn = connect(db_path)
    run_id, failures = _capture_run(
        conn, kind="baseline", prompts=prompts, provider=provider
    )
    captured = len(prompts) - failures
    print(
        f"Baseline complete: {captured}/{len(prompts)} prompts captured "
        f"with provider={provider.name} model={provider.chat_model}. run_id={run_id}"
    )
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
    prompts_path: Path,
    db_path: Path,
    threshold: float,
    report_dir: Path,
    provider_name: str | None,
) -> int:
    prompts = _load_prompts(prompts_path)

    if not db_path.exists():
        print("No baseline run found. Run 'drift baseline' first.")
        return 1

    conn = connect(db_path)
    baseline = latest_baseline_run(conn)
    if baseline is None:
        print("No baseline run found. Run 'drift baseline' first.")
        return 1

    if provider_name is None:
        provider_name = baseline["provider"]
    elif provider_name != baseline["provider"]:
        print(
            f"Cannot compare: latest baseline used '{baseline['provider']}', "
            f"requested provider is '{provider_name}'."
        )
        return 1

    provider = get_provider(provider_name)

    eval_run_id, failures = _capture_run(
        conn, kind="eval", prompts=prompts, provider=provider
    )

    baseline_responses = responses_for_run(conn, baseline["id"])
    eval_responses = responses_for_run(conn, eval_run_id)
    comparisons = _build_comparisons(baseline_responses, eval_responses, threshold)

    eval_run_row = conn.execute(
        "SELECT id, started_at, model, embedding_model, kind, provider "
        "FROM runs WHERE id = ?",
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
