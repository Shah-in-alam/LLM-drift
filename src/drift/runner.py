import sqlite3
from collections import defaultdict
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from drift.metrics import centroid, cosine, intra_set_avg_cosine
from drift.notifications.slack import FailedPromptSummary, notify_drift
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
    samples: int,
    temperature: float,
) -> tuple[int, int]:
    """Run the prompt suite once, store N samples per prompt. Returns (run_id, failure_count)."""
    embed = get_embedder()
    run_id = insert_run(
        conn,
        model=provider.chat_model,
        embedding_model=EMBEDDING_MODEL,
        kind=kind,
        provider=provider.name,
        samples=samples,
        temperature=temperature,
    )
    failures = 0
    for p in prompts:
        for i in range(samples):
            try:
                response = provider.chat(p.text, temperature=temperature)
                embedding = embed(response)
            except Exception as exc:
                failures += 1
                print(f"[fail] prompt_id={p.id} sample={i}: {exc}")
                continue

            insert_response(
                conn,
                run_id=run_id,
                prompt_id=p.id,
                prompt_text=p.text,
                response_text=response,
                embedding=embedding,
                sample_idx=i,
            )
            print(f"[ok] prompt_id={p.id} sample={i} ({len(response)} chars, {len(embedding)}-dim)")
    return run_id, failures


def run_baseline(
    prompts_path: Path,
    db_path: Path,
    provider_name: str,
    samples: int,
    temperature: float,
) -> int:
    provider = get_provider(provider_name)
    prompts = _load_prompts(prompts_path)
    conn = connect(db_path)
    run_id, failures = _capture_run(
        conn,
        kind="baseline",
        prompts=prompts,
        provider=provider,
        samples=samples,
        temperature=temperature,
    )
    captured = len(prompts) * samples - failures
    expected = len(prompts) * samples
    print(
        f"Baseline complete: {captured}/{expected} responses captured "
        f"with provider={provider.name} model={provider.chat_model} "
        f"samples={samples} temperature={temperature}. run_id={run_id}"
    )
    return 1 if failures else 0


def _group_by_prompt(responses: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in responses:
        groups[r["prompt_id"]].append(r)
    return groups


def _build_comparisons(
    baseline_responses: list[dict],
    eval_responses: list[dict],
    threshold: float,
) -> list[Comparison]:
    baseline_by_pid = _group_by_prompt(baseline_responses)
    eval_by_pid = _group_by_prompt(eval_responses)

    # Preserve eval insertion order for prompt sequencing in the report.
    eval_pids_in_order: list[str] = []
    seen: set[str] = set()
    for r in eval_responses:
        if r["prompt_id"] not in seen:
            eval_pids_in_order.append(r["prompt_id"])
            seen.add(r["prompt_id"])

    comparisons: list[Comparison] = []

    for pid in eval_pids_in_order:
        eval_samples = eval_by_pid[pid]
        if pid in baseline_by_pid:
            baseline_samples = baseline_by_pid[pid]
            b_embeddings = [s["embedding"] for s in baseline_samples]
            e_embeddings = [s["embedding"] for s in eval_samples]
            sim = cosine(centroid(b_embeddings), centroid(e_embeddings))
            comparisons.append(
                Comparison(
                    prompt_id=pid,
                    kind="compared",
                    similarity=sim,
                    baseline_response=baseline_samples[0]["response_text"],
                    eval_response=eval_samples[0]["response_text"],
                    passed=sim >= threshold,
                    n_baseline=len(baseline_samples),
                    n_eval=len(eval_samples),
                    baseline_noise=intra_set_avg_cosine(b_embeddings),
                )
            )
        else:
            comparisons.append(
                Comparison(
                    prompt_id=pid,
                    kind="new",
                    similarity=None,
                    baseline_response=None,
                    eval_response=eval_samples[0]["response_text"],
                    passed=None,
                    n_baseline=0,
                    n_eval=len(eval_samples),
                    baseline_noise=None,
                )
            )

    for pid, baseline_samples in baseline_by_pid.items():
        if pid not in eval_by_pid:
            comparisons.append(
                Comparison(
                    prompt_id=pid,
                    kind="missing",
                    similarity=None,
                    baseline_response=baseline_samples[0]["response_text"],
                    eval_response=None,
                    passed=None,
                    n_baseline=len(baseline_samples),
                    n_eval=0,
                    baseline_noise=None,
                )
            )
    return comparisons


def run_eval(
    prompts_path: Path,
    db_path: Path,
    threshold: float,
    report_dir: Path,
    provider_name: str | None,
    samples: int | None,
    temperature: float | None,
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

    if samples is None:
        samples = int(baseline["samples"])
    if temperature is None:
        temperature = float(baseline["temperature"])

    provider = get_provider(provider_name)

    eval_run_id, failures = _capture_run(
        conn,
        kind="eval",
        prompts=prompts,
        provider=provider,
        samples=samples,
        temperature=temperature,
    )

    baseline_responses = responses_for_run(conn, baseline["id"])
    eval_responses = responses_for_run(conn, eval_run_id)
    comparisons = _build_comparisons(baseline_responses, eval_responses, threshold)

    eval_run_row = conn.execute(
        "SELECT id, started_at, model, embedding_model, kind, provider, samples, temperature "
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

    failed_compared = [c for c in comparisons if c.kind == "compared" and not c.passed]
    total_compared = sum(1 for c in comparisons if c.kind == "compared")
    result = "FAIL" if failed_compared else "PASS"
    print(f"Drift report: {result} — wrote {report_path}")

    if failed_compared:
        notify_drift(
            eval_run_id=eval_run_id,
            baseline_run_id=int(baseline["id"]),
            provider=provider.name,
            model=provider.chat_model,
            threshold=threshold,
            failed_count=len(failed_compared),
            total_compared=total_compared,
            failed_prompts=[
                FailedPromptSummary(
                    prompt_id=c.prompt_id,
                    similarity=c.similarity if c.similarity is not None else 0.0,
                    baseline_excerpt=c.baseline_response or "",
                    eval_excerpt=c.eval_response or "",
                )
                for c in failed_compared
            ],
            report_path=str(report_path),
        )

    return 1 if (failed_compared or failures) else 0
