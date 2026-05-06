from dataclasses import dataclass


@dataclass(frozen=True)
class Comparison:
    prompt_id: str
    kind: str  # "compared" | "new" | "missing"
    similarity: float | None  # only set when kind == "compared"
    baseline_response: str | None
    eval_response: str | None
    passed: bool | None  # only set when kind == "compared"


def build_markdown(
    *,
    eval_run: dict,
    baseline_run: dict,
    comparisons: list[Comparison],
    threshold: float,
) -> str:
    compared = [c for c in comparisons if c.kind == "compared"]
    failed = [c for c in compared if not c.passed]
    result = "FAIL" if failed else "PASS"

    lines = [
        f"# Drift report — run {eval_run['id']} (eval) vs run {baseline_run['id']} (baseline)",
        "",
        f"- date: {eval_run['started_at']}",
        f"- provider: {eval_run['provider']} ({eval_run['model']})",
        f"- embedding: {eval_run['embedding_model']}",
        f"- threshold: {threshold}",
        f"- result: {result} ({len(failed)}/{len(compared)} prompts below threshold)",
        "",
    ]

    # eval-order first (compared + new), then missing at the bottom
    primary = [c for c in comparisons if c.kind in ("compared", "new")]
    missing = [c for c in comparisons if c.kind == "missing"]

    for c in primary:
        if c.kind == "compared":
            mark = "✓" if c.passed else "✗"
            lines.append(f"## {c.prompt_id} — sim {c.similarity:.3f} {mark}")
            lines.append(f"**Baseline:** {c.baseline_response}")
            lines.append(f"**Now:** {c.eval_response}")
        else:  # new
            lines.append(f"## {c.prompt_id} — [new]")
            lines.append(f"**Now:** {c.eval_response}")
        lines.append("")

    for c in missing:
        lines.append(f"## {c.prompt_id} — [missing]")
        lines.append(f"**Baseline:** {c.baseline_response}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
