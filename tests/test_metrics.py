import math

import pytest

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


def test_identical_vectors():
    assert math.isclose(cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 1.0)


def test_orthogonal_vectors():
    assert math.isclose(cosine([1.0, 0.0], [0.0, 1.0]), 0.0)


def test_negated_vectors():
    assert math.isclose(cosine([1.0, 2.0], [-1.0, -2.0]), -1.0)


def test_zero_vector_returns_zero():
    assert cosine([0.0, 0.0], [1.0, 2.0]) == 0.0


def test_centroid_averages_vectors():
    result = centroid([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert result == [3.0, 4.0]


def test_centroid_rejects_empty():
    with pytest.raises(ValueError):
        centroid([])


def test_intra_set_avg_cosine_single_vector_is_one():
    assert intra_set_avg_cosine([[1.0, 2.0, 3.0]]) == 1.0


def test_intra_set_avg_cosine_identical_pair_is_one():
    assert math.isclose(intra_set_avg_cosine([[1.0, 0.0], [1.0, 0.0]]), 1.0)


def test_intra_set_avg_cosine_orthogonal_pair_is_zero():
    assert math.isclose(intra_set_avg_cosine([[1.0, 0.0], [0.0, 1.0]]), 0.0, abs_tol=1e-9)


def test_histogram_normalizes_to_one():
    h = histogram([0.05, 0.15, 0.55, 0.95], bins=10)
    assert len(h) == 10
    assert math.isclose(sum(h), 1.0)


def test_histogram_empty_input_returns_zeros():
    assert histogram([], bins=5) == [0.0] * 5


def test_psi_identical_distributions_is_zero():
    p = [0.1, 0.2, 0.3, 0.2, 0.2]
    assert math.isclose(psi(p, p), 0.0, abs_tol=1e-9)


def test_psi_signals_shift():
    p = [0.0, 0.0, 0.0, 0.0, 1.0]  # all probability mass in last bin
    q = [1.0, 0.0, 0.0, 0.0, 0.0]  # all in first bin
    # PSI for a complete shift between two well-separated distributions is large.
    assert psi(p, q) > 1.0


def test_psi_mismatched_shapes_raises():
    with pytest.raises(ValueError):
        psi([0.5, 0.5], [0.3, 0.3, 0.4])


def test_kl_divergence_identical_is_zero():
    p = [0.25, 0.25, 0.25, 0.25]
    assert math.isclose(kl_divergence(p, p), 0.0, abs_tol=1e-9)


def test_kl_divergence_asymmetric():
    # KL(p||q) is not symmetric in general.
    p = [0.9, 0.1]
    q = [0.5, 0.5]
    assert kl_divergence(p, q) != kl_divergence(q, p)


def test_token_edit_distance_identical_is_zero():
    assert token_edit_distance("the answer is 42", "the answer is 42") == 0.0


def test_token_edit_distance_both_empty_is_zero():
    assert token_edit_distance("", "") == 0.0


def test_token_edit_distance_one_empty_is_one():
    assert token_edit_distance("hello world", "") == 1.0
    assert token_edit_distance("", "hello world") == 1.0


def test_token_edit_distance_one_word_diff_small():
    # 1 substitution out of 4 tokens → 0.25.
    assert math.isclose(token_edit_distance("the answer is 42", "the answer is 43"), 0.25)


def test_token_edit_distance_added_format_tokens_signals_drift():
    # Cosine semantics would say "same answer" (0.99-ish), but adding 4 leading
    # format tokens around the kept "611" yields a large normalized edit distance.
    base = "611"
    bullet_paraphrase = "* The answer is 611"
    # 1 token vs 5 tokens, sharing only "611" → 4 insertions / max(1,5) = 0.8.
    assert math.isclose(token_edit_distance(base, bullet_paraphrase), 0.8)
    # Should comfortably trip the default 0.3 edit threshold.
    assert token_edit_distance(base, bullet_paraphrase) > 0.3


def test_token_edit_distance_normalized_to_unit_interval():
    d = token_edit_distance("hello there friend", "hi there pal")
    assert 0.0 <= d <= 1.0


def test_token_edit_distance_handles_extra_whitespace():
    assert token_edit_distance("a b  c", " a   b c") == 0.0


def test_drift_significance_detects_clear_drift():
    """Eval responses are clearly different from baseline → low p, positive effect."""
    # Baseline: tightly clustered around (1.0, 0.0)
    baseline = [
        [1.0, 0.01],
        [1.0, -0.01],
        [0.99, 0.02],
        [1.0, 0.0],
        [0.99, -0.02],
    ]
    # Eval: rotated to (0.0, 1.0) — very different
    eval_ = [
        [0.01, 1.0],
        [-0.01, 1.0],
        [0.0, 0.99],
        [0.02, 1.0],
        [-0.02, 0.99],
    ]
    result = drift_significance(baseline, eval_)
    assert result is not None
    effect, p = result
    assert effect > 0.5, f"expected large positive effect, got {effect}"
    assert p < 0.05, f"expected significant p, got {p}"


def test_drift_significance_no_drift_when_eval_matches_baseline():
    """Eval is drawn from the same distribution as baseline → not significant."""
    baseline = [
        [1.0, 0.01],
        [1.0, -0.01],
        [0.99, 0.02],
        [1.0, 0.0],
        [0.99, -0.02],
    ]
    # Eval samples from the same neighborhood — no drift
    eval_ = [
        [1.0, 0.0],
        [0.99, 0.01],
        [1.0, -0.02],
    ]
    result = drift_significance(baseline, eval_)
    assert result is not None
    effect, p = result
    # Effect should be tiny in either direction.
    assert abs(effect) < 0.05
    # p should not be tiny — we're not seeing a real shift.
    assert p > 0.05


def test_drift_significance_degenerate_with_single_baseline_sample():
    # Baseline self-cosine distribution is empty when n < 2.
    assert drift_significance([[1.0, 0.0]], [[1.0, 0.0], [1.0, 0.01]]) is None


def test_drift_significance_degenerate_with_no_eval_samples():
    assert drift_significance([[1.0, 0.0], [1.0, 0.01]], []) is None
