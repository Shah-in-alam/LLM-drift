import sqlite3
from collections import defaultdict
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from drift.metrics import (
    centroid,
    cosine,
    drift_significance,
    histogram,
    intra_set_avg_cosine,
    kl_divergence,
    psi,
    token_edit_distance,
)
from drift.notifications.slack import FailedPromptSummary, notify_drift
from drift.providers import ChatProvider, get_embedder, get_provider
from drift.providers.openai_embed import EMBEDDING_MODEL
from drift.report import Comparison, RollingMetric, build_markdown
from drift.storage import (
    connect,
    insert_response,
    insert_run,
    latest_baseline_run,
    recent_eval_runs,
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


def _mean_pairwise_edit(baseline_texts: list[str], eval_texts: list[str]) -> float:
    distances = [token_edit_distance(b, e) for b in baseline_texts for e in eval_texts]
    return sum(distances) / len(distances) if distances else 0.0


def _build_comparisons(
    baseline_responses: list[dict],
    eval_responses: list[dict],
    threshold: float,
    edit_threshold: float,
    p_threshold: float,
    effect_delta: float,
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
            edit = _mean_pairwise_edit(
                [s["response_text"] for s in baseline_samples],
                [s["response_text"] for s in eval_samples],
            )
            sig_result = drift_significance(b_embeddings, e_embeddings)
            if sig_result is None:
                effect, p_val, significant = None, None, None
            else:
                effect, p_val = sig_result
                significant = (effect > effect_delta) and (p_val < p_threshold)
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
                    edit_distance=edit,
                    edit_passed=edit < edit_threshold,
                    p_value=p_val,
                    effect_size=effect,
                    significant_drift=significant,
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


def _build_rolling_metrics(
    conn: sqlite3.Connection,
    *,
    baseline_responses: list[dict],
    rolling_window: int,
    psi_threshold: float,
) -> dict[str, RollingMetric]:
    """Compute per-prompt PSI / KL between baseline and the recent eval-run window.

    Degenerate when baseline has <2 samples or there's no eval data for the prompt.
    """
    baseline_by_pid = _group_by_prompt(baseline_responses)
    recent = recent_eval_runs(conn, limit=rolling_window)
    rolling_responses: list[dict] = []
    for run in recent:
        rolling_responses.extend(responses_for_run(conn, int(run["id"])))
    rolling_by_pid = _group_by_prompt(rolling_responses)

    # How many of the recent runs included this prompt at all
    runs_per_pid: dict[str, set[int]] = defaultdict(set)
    for r in rolling_responses:
        runs_per_pid[r["prompt_id"]].add(int(r["run_id"]))

    metrics: dict[str, RollingMetric] = {}
    for pid, baseline_samples in baseline_by_pid.items():
        n_runs = len(runs_per_pid.get(pid, set()))

        if len(baseline_samples) < 2 or pid not in rolling_by_pid:
            metrics[pid] = RollingMetric(
                prompt_id=pid,
                psi=None,
                kl=None,
                psi_passed=None,
                n_runs=n_runs,
            )
            continue

        b_embs = [s["embedding"] for s in baseline_samples]
        baseline_pairs = [
            cosine(b_embs[i], b_embs[j])
            for i in range(len(b_embs))
            for j in range(i + 1, len(b_embs))
        ]

        rolling_pairs = [cosine(s["embedding"], b) for s in rolling_by_pid[pid] for b in b_embs]

        p = histogram(baseline_pairs)
        q = histogram(rolling_pairs)
        psi_val = psi(p, q)
        kl_val = kl_divergence(p, q)
        metrics[pid] = RollingMetric(
            prompt_id=pid,
            psi=psi_val,
            kl=kl_val,
            psi_passed=psi_val < psi_threshold,
            n_runs=n_runs,
        )
    return metrics


def run_eval(
    prompts_path: Path,
    db_path: Path,
    threshold: float,
    report_dir: Path,
    provider_name: str | None,
    samples: int | None,
    temperature: float | None,
    psi_threshold: float = 0.25,
    rolling_window: int = 7,
    edit_threshold: float = 0.3,
    p_threshold: float = 0.05,
    effect_delta: float = 0.01,
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
    comparisons = _build_comparisons(
        baseline_responses,
        eval_responses,
        threshold,
        edit_threshold,
        p_threshold,
        effect_delta,
    )
    rolling = _build_rolling_metrics(
        conn,
        baseline_responses=baseline_responses,
        rolling_window=rolling_window,
        psi_threshold=psi_threshold,
    )

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
        rolling=rolling,
        psi_threshold=psi_threshold,
        rolling_window=rolling_window,
    )

    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"run-{eval_run_id}.md"
    report_path.write_text(markdown, encoding="utf-8")

    failed_compared = [c for c in comparisons if c.kind == "compared" and not c.passed]
    edit_failed = [c for c in comparisons if c.kind == "compared" and c.edit_passed is False]
    total_compared = sum(1 for c in comparisons if c.kind == "compared")
    psi_failed = [m for m in rolling.values() if m.psi_passed is False]
    has_drift = bool(failed_compared or edit_failed or psi_failed)
    result = "FAIL" if has_drift else "PASS"
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

    return 1 if (has_drift or failures) else 0
