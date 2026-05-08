from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

_TIMEOUT_SECONDS = 10
_MAX_FAILED_PROMPTS_IN_MESSAGE = 3
_EXCERPT_LIMIT = 120


@dataclass(frozen=True)
class FailedPromptSummary:
    prompt_id: str
    similarity: float
    baseline_excerpt: str
    eval_excerpt: str


def _excerpt(text: str, limit: int = _EXCERPT_LIMIT) -> str:
    flat = text.replace("\n", " ").strip()
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _build_blocks(
    *,
    eval_run_id: int,
    baseline_run_id: int,
    provider: str,
    model: str,
    threshold: float,
    failed_count: int,
    total_compared: int,
    failed_prompts: list[FailedPromptSummary],
    report_path: str,
) -> list[dict]:
    blocks: list[dict] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"⚠️  LLM drift detected — run {eval_run_id}",
                "emoji": True,
            },
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Failed:*\n{failed_count}/{total_compared}"},
                {"type": "mrkdwn", "text": f"*Threshold:*\n{threshold}"},
                {"type": "mrkdwn", "text": f"*Provider:*\n{provider}"},
                {"type": "mrkdwn", "text": f"*Model:*\n{model}"},
                {"type": "mrkdwn", "text": f"*Baseline:*\nrun {baseline_run_id}"},
                {"type": "mrkdwn", "text": f"*Report:*\n`{report_path}`"},
            ],
        },
    ]

    for p in failed_prompts[:_MAX_FAILED_PROMPTS_IN_MESSAGE]:
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*{p.prompt_id}* — sim `{p.similarity:.3f}`\n"
                        f"> *Baseline:* {_excerpt(p.baseline_excerpt)}\n"
                        f"> *Now:* {_excerpt(p.eval_excerpt)}"
                    ),
                },
            }
        )

    if len(failed_prompts) > _MAX_FAILED_PROMPTS_IN_MESSAGE:
        remaining = len(failed_prompts) - _MAX_FAILED_PROMPTS_IN_MESSAGE
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"_…and {remaining} more failed prompts in the report._",
                    }
                ],
            }
        )

    return blocks


def notify_drift(
    *,
    eval_run_id: int,
    baseline_run_id: int,
    provider: str,
    model: str,
    threshold: float,
    failed_count: int,
    total_compared: int,
    failed_prompts: list[FailedPromptSummary],
    report_path: str,
) -> None:
    """Post a Slack message when SLACK_WEBHOOK_URL is set; otherwise silently no-op.

    Never raises — Slack failures print a warning and return so a flaky webhook
    cannot crash the eval pipeline.
    """
    webhook = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook:
        return

    blocks = _build_blocks(
        eval_run_id=eval_run_id,
        baseline_run_id=baseline_run_id,
        provider=provider,
        model=model,
        threshold=threshold,
        failed_count=failed_count,
        total_compared=total_compared,
        failed_prompts=failed_prompts,
        report_path=report_path,
    )

    payload = json.dumps(
        {
            "text": (
                f"Drift detected on run {eval_run_id}: "
                f"{failed_count}/{total_compared} prompts below {threshold}"
            ),
            "blocks": blocks,
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        webhook,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as resp:
            if resp.status != 200:
                print(f"[warn] Slack returned status {resp.status}")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"[warn] Slack notification failed: {exc}")
