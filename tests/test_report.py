from drift.report import Comparison, RollingMetric, build_markdown


def _runs():
    eval_run = {
        "id": 7,
        "started_at": "2026-05-06T12:00:00+00:00",
        "model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-large",
        "provider": "openai",
        "samples": 1,
        "temperature": 0.0,
    }
    baseline_run = {
        "id": 1,
        "started_at": "2026-05-05T09:00:00+00:00",
        "model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-large",
        "provider": "openai",
        "samples": 1,
        "temperature": 0.0,
    }
    return eval_run, baseline_run


def test_markdown_fail_with_mixed_comparisons():
    eval_run, baseline_run = _runs()
    comparisons = [
        Comparison("greet", "compared", 0.987, "Hello!", "Hi there!", True, 1, 1, 1.0),
        Comparison("math", "compared", 0.823, "611", "13 × 47 = 611", False, 1, 1, 1.0),
        Comparison("brand_new", "new", None, None, "Some new response", None, 0, 1, None),
        Comparison("retired", "missing", None, "Old response", None, None, 1, 0, None),
    ]

    md = build_markdown(
        eval_run=eval_run,
        baseline_run=baseline_run,
        comparisons=comparisons,
        threshold=0.95,
    )

    assert "# Drift report — run 7 (eval) vs run 1 (baseline)" in md
    assert "- provider: openai (gpt-4o-mini)" in md
    assert "- threshold: 0.95" in md
    assert (
        "result: FAIL (1/2 below cosine threshold, "
        "0 above edit threshold, 0 above PSI threshold)" in md
    )
    assert "## greet — sim 0.987 ✓" in md
    assert "## math — sim 0.823 ✗" in md
    assert "## brand_new — [new]" in md
    assert "## retired — [missing]" in md
    # missing section comes after [new] / compared
    assert md.index("## retired") > md.index("## brand_new")


def test_markdown_pass_when_all_above_threshold():
    eval_run, baseline_run = _runs()
    comparisons = [
        Comparison("a", "compared", 0.99, "x", "x", True, 1, 1, 1.0),
        Comparison("b", "compared", 0.98, "y", "y", True, 1, 1, 1.0),
    ]
    md = build_markdown(
        eval_run=eval_run,
        baseline_run=baseline_run,
        comparisons=comparisons,
        threshold=0.95,
    )
    assert (
        "result: PASS (0/2 below cosine threshold, "
        "0 above edit threshold, 0 above PSI threshold)" in md
    )


def test_markdown_includes_sample_metadata():
    eval_run, baseline_run = _runs()
    eval_run["samples"] = 3
    eval_run["temperature"] = 0.7
    baseline_run["samples"] = 3
    baseline_run["temperature"] = 0.7
    comparisons = [
        Comparison("a", "compared", 0.96, "x", "x", True, 3, 3, 0.992),
    ]
    md = build_markdown(
        eval_run=eval_run,
        baseline_run=baseline_run,
        comparisons=comparisons,
        threshold=0.95,
    )
    assert "samples: baseline=3 (temp=0.7), eval=3 (temp=0.7)" in md
    assert "n_baseline=3, n_eval=3" in md
    assert "baseline_noise=0.992" in md


def test_markdown_renders_rolling_psi_pass():
    eval_run, baseline_run = _runs()
    eval_run["samples"] = 3
    baseline_run["samples"] = 3
    comparisons = [
        Comparison("greet", "compared", 0.99, "Hi", "Hi!", True, 3, 3, 0.99),
    ]
    rolling = {
        "greet": RollingMetric(prompt_id="greet", psi=0.04, kl=0.01, psi_passed=True, n_runs=3),
    }
    md = build_markdown(
        eval_run=eval_run,
        baseline_run=baseline_run,
        comparisons=comparisons,
        threshold=0.95,
        rolling=rolling,
        psi_threshold=0.25,
        rolling_window=7,
    )
    assert "rolling window: last 7 eval run(s), PSI threshold 0.25" in md
    assert "rolling psi 0.040 ✓" in md
    assert "kl 0.010 over 3 eval run(s)" in md


def test_markdown_renders_rolling_psi_fail_flips_overall_result():
    eval_run, baseline_run = _runs()
    eval_run["samples"] = 3
    baseline_run["samples"] = 3
    comparisons = [
        Comparison("greet", "compared", 0.99, "Hi", "Hi!", True, 3, 3, 0.99),
    ]
    rolling = {
        "greet": RollingMetric(prompt_id="greet", psi=0.50, kl=0.30, psi_passed=False, n_runs=5),
    }
    md = build_markdown(
        eval_run=eval_run,
        baseline_run=baseline_run,
        comparisons=comparisons,
        threshold=0.95,
        rolling=rolling,
        psi_threshold=0.25,
        rolling_window=7,
    )
    # Cosine passed but PSI failed → overall FAIL with 1 above PSI threshold.
    assert (
        "result: FAIL (0/1 below cosine threshold, "
        "0 above edit threshold, 1 above PSI threshold)" in md
    )
    assert "rolling psi 0.500 ✗" in md


def test_markdown_renders_degenerate_rolling_when_baseline_singleton():
    eval_run, baseline_run = _runs()
    comparisons = [
        Comparison("greet", "compared", 0.99, "Hi", "Hi!", True, 1, 1, 1.0),
    ]
    rolling = {
        "greet": RollingMetric(prompt_id="greet", psi=None, kl=None, psi_passed=None, n_runs=1),
    }
    md = build_markdown(
        eval_run=eval_run,
        baseline_run=baseline_run,
        comparisons=comparisons,
        threshold=0.95,
        rolling=rolling,
        psi_threshold=0.25,
        rolling_window=7,
    )
    assert "rolling: not computed (baseline needs >=2 samples)" in md


def test_markdown_renders_edit_distance_pass():
    eval_run, baseline_run = _runs()
    comparisons = [
        Comparison(
            prompt_id="greet",
            kind="compared",
            similarity=0.99,
            baseline_response="Hi!",
            eval_response="Hello!",
            passed=True,
            n_baseline=1,
            n_eval=1,
            baseline_noise=1.0,
            edit_distance=0.05,
            edit_passed=True,
        ),
    ]
    md = build_markdown(
        eval_run=eval_run,
        baseline_run=baseline_run,
        comparisons=comparisons,
        threshold=0.95,
    )
    assert "sim 0.990 ✓, edit 0.050 ✓" in md


def test_markdown_renders_edit_distance_fail_flips_overall_result():
    eval_run, baseline_run = _runs()
    comparisons = [
        # Cosine PASSES (semantically the same answer) but edit FAILS
        # because the model started prepending bullets.
        Comparison(
            prompt_id="math",
            kind="compared",
            similarity=0.99,
            baseline_response="611",
            eval_response="* The answer is 611",
            passed=True,
            n_baseline=1,
            n_eval=1,
            baseline_noise=1.0,
            edit_distance=0.85,
            edit_passed=False,
        ),
    ]
    md = build_markdown(
        eval_run=eval_run,
        baseline_run=baseline_run,
        comparisons=comparisons,
        threshold=0.95,
    )
    assert (
        "result: FAIL (0/1 below cosine threshold, "
        "1 above edit threshold, 0 above PSI threshold)" in md
    )
    assert "sim 0.990 ✓, edit 0.850 ✗" in md
