from dataclasses import dataclass


@dataclass(frozen=True)
class Comparison:
    prompt_id: str
    kind: str  # "compared" | "new" | "missing"
    similarity: float | None  # cosine of (baseline centroid, eval centroid); only when "compared"
    baseline_response: str | None  # one example, for human readability
    eval_response: str | None
    passed: bool | None  # cosine threshold check; only when "compared"
    n_baseline: int = 0  # number of baseline samples
    n_eval: int = 0  # number of eval samples
    baseline_noise: float | None = None  # avg pairwise cosine within baseline (1.0 if n=1)
    edit_distance: float | None = None  # mean normalized token-edit distance, [0,1]
    edit_passed: bool | None = None  # edit_distance < edit_threshold
    p_value: float | None = None  # Mann-Whitney U: signal stochastically lower than noise
    effect_size: float | None = None  # mean(noise) - mean(signal); positive = drift direction
    significant_drift: bool | None = None  # effect > delta AND p < alpha


@dataclass(frozen=True)
class RollingMetric:
    """Distribution-level drift metrics for a prompt over the recent eval-run window."""

    prompt_id: str
    psi: float | None  # None = degenerate (baseline samples < 2 or no eval data)
    kl: float | None
    psi_passed: bool | None
    n_runs: int  # how many eval runs contributed to the window for this prompt


def build_markdown(
    *,
    eval_run: dict,
    baseline_run: dict,
    comparisons: list[Comparison],
    threshold: float,
    rolling: dict[str, RollingMetric] | None = None,
    psi_threshold: float | None = None,
    rolling_window: int | None = None,
) -> str:
    rolling = rolling or {}
    compared = [c for c in comparisons if c.kind == "compared"]
    cosine_failed = [c for c in compared if not c.passed]
    edit_failed = [c for c in compared if c.edit_passed is False]
    psi_failed = [m for m in rolling.values() if m.psi_passed is False]
    result = "FAIL" if (cosine_failed or edit_failed or psi_failed) else "PASS"

    lines = [
        f"# Drift report — run {eval_run['id']} (eval) vs run {baseline_run['id']} (baseline)",
        "",
        f"- date: {eval_run['started_at']}",
        f"- provider: {eval_run['provider']} ({eval_run['model']})",
        f"- embedding: {eval_run['embedding_model']}",
        f"- samples: baseline={baseline_run['samples']} (temp={baseline_run['temperature']}), "
        f"eval={eval_run['samples']} (temp={eval_run['temperature']})",
        f"- threshold: {threshold}",
    ]
    if psi_threshold is not None and rolling_window is not None:
        lines.append(
            f"- rolling window: last {rolling_window} eval run(s), PSI threshold {psi_threshold}"
        )
    lines += [
        (
            f"- result: {result} "
            f"({len(cosine_failed)}/{len(compared)} below cosine threshold, "
            f"{len(edit_failed)} above edit threshold, "
            f"{len(psi_failed)} above PSI threshold)"
        ),
        "",
    ]

    primary = [c for c in comparisons if c.kind in ("compared", "new")]
    missing = [c for c in comparisons if c.kind == "missing"]

    for c in primary:
        if c.kind == "compared":
            cosine_mark = "✓" if c.passed else "✗"
            extras = (
                f"n_baseline={c.n_baseline}, n_eval={c.n_eval}, "
                f"baseline_noise={c.baseline_noise:.3f}"
                if c.baseline_noise is not None
                else f"n_baseline={c.n_baseline}, n_eval={c.n_eval}"
            )
            head = f"## {c.prompt_id} — sim {c.similarity:.3f} {cosine_mark}"
            if c.edit_distance is not None:
                edit_mark = "✓" if c.edit_passed else "✗"
                head += f", edit {c.edit_distance:.3f} {edit_mark}"
            head += f" ({extras})"
            lines.append(head)

            if c.p_value is not None:
                conf = "significant" if c.significant_drift else "not significant"
                lines.append(
                    f"- significance: p={c.p_value:.3g}, effect={c.effect_size:.3f}, {conf}"
                )

            metric = rolling.get(c.prompt_id)
            if metric is not None:
                if metric.psi is None:
                    lines.append("- rolling: not computed (baseline needs >=2 samples)")
                else:
                    psi_mark = "✓" if metric.psi_passed else "✗"
                    lines.append(
                        f"- rolling psi {metric.psi:.3f} {psi_mark}, "
                        f"kl {metric.kl:.3f} over {metric.n_runs} eval run(s)"
                    )

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
