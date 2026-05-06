from drift.report import Comparison, build_markdown


def _runs():
    eval_run = {
        "id": 7,
        "started_at": "2026-05-06T12:00:00+00:00",
        "model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-large",
    }
    baseline_run = {
        "id": 1,
        "started_at": "2026-05-05T09:00:00+00:00",
        "model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-large",
    }
    return eval_run, baseline_run


def test_markdown_fail_with_mixed_comparisons():
    eval_run, baseline_run = _runs()
    comparisons = [
        Comparison("greet", "compared", 0.987, "Hello!", "Hi there!", True),
        Comparison("math", "compared", 0.823, "611", "13 × 47 = 611", False),
        Comparison("brand_new", "new", None, None, "Some new response", None),
        Comparison("retired", "missing", None, "Old response", None, None),
    ]

    md = build_markdown(
        eval_run=eval_run,
        baseline_run=baseline_run,
        comparisons=comparisons,
        threshold=0.95,
    )

    assert "# Drift report — run 7 (eval) vs run 1 (baseline)" in md
    assert "- threshold: 0.95" in md
    assert "result: FAIL (1/2 prompts below threshold)" in md
    assert "## greet — sim 0.987 ✓" in md
    assert "## math — sim 0.823 ✗" in md
    assert "## brand_new — [new]" in md
    assert "## retired — [missing]" in md
    # missing section comes after [new] / compared
    assert md.index("## retired") > md.index("## brand_new")


def test_markdown_pass_when_all_above_threshold():
    eval_run, baseline_run = _runs()
    comparisons = [
        Comparison("a", "compared", 0.99, "x", "x", True),
        Comparison("b", "compared", 0.98, "y", "y", True),
    ]
    md = build_markdown(
        eval_run=eval_run,
        baseline_run=baseline_run,
        comparisons=comparisons,
        threshold=0.95,
    )
    assert "result: PASS (0/2 prompts below threshold)" in md
