"""
End-to-end demo of `drift baseline` and `drift run` with fake provider + embedder.

This bypasses the OpenAI / Anthropic SDKs entirely so we can show the pipeline
working without API keys. It is NOT part of the product — it's a demo harness.
"""

from __future__ import annotations

import hashlib
import math
import shutil
import sys
from pathlib import Path

# Deterministic fake responses keyed by prompt id, with two "snapshots" to
# simulate model drift between baseline (snapshot=1) and eval (snapshot=2).
_FAKE_RESPONSES = {
    "greet": {
        1: "Hello!",
        2: "Hi there! Hope you're well.",  # mild drift — chat model got chattier
    },
    "math": {
        1: "611",
        2: "611",  # identical — should sim ~1.0
    },
    "refusal": {
        1: "I won't write malware because it harms people and is illegal.",
        2: "Writing malware is harmful and unlawful, so I decline.",  # paraphrase
    },
}


def _fake_embed(text: str, dim: int = 128) -> list[float]:
    """Deterministic 'embedding' that's similar for similar substrings.

    Real embeddings have semantic structure; we approximate by mixing a
    bag-of-character-trigrams signal with a word-level signal so paraphrases
    of the same concept land near each other but identical strings collapse
    to the same vector.
    """
    vec = [0.0] * dim
    # character trigrams
    s = text.lower()
    for i in range(len(s) - 2):
        h = int(hashlib.sha1(s[i : i + 3].encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    # words for some semantic signal
    for word in s.split():
        h = int(hashlib.sha1(word.encode()).hexdigest(), 16)
        vec[h % dim] += 2.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class FakeChatProvider:
    name = "openai"
    chat_model = "gpt-4o-mini"

    def __init__(self, snapshot: int) -> None:
        self.snapshot = snapshot

    def chat(self, prompt: str, *, temperature: float = 0.0) -> str:
        del temperature  # mock ignores; kept only to match ChatProvider protocol
        for pid, snaps in _FAKE_RESPONSES.items():
            if prompt.startswith(("Say hello", "What is 13", "Briefly explain")):
                if (
                    (pid == "greet" and "hello" in prompt.lower())
                    or (pid == "math" and "13" in prompt)
                    or (pid == "refusal" and "malware" in prompt)
                ):
                    return snaps[self.snapshot]
        return f"[fake response to: {prompt}]"


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    db_path = repo / "drift.db"
    reports = repo / "reports"
    if db_path.exists():
        db_path.unlink()
    if reports.exists():
        shutil.rmtree(reports)

    # Patch the provider factory + embedder before importing the runner.
    import drift.providers as providers_pkg

    snapshot_holder = {"n": 1}

    def fake_get_provider(name: str):
        return FakeChatProvider(snapshot=snapshot_holder["n"])

    providers_pkg.get_provider = fake_get_provider  # type: ignore[assignment]
    providers_pkg.get_embedder = lambda: _fake_embed  # type: ignore[assignment]

    # Re-import runner so it picks up the patched factories.
    import importlib

    import drift.runner

    importlib.reload(drift.runner)
    from drift.runner import run_baseline, run_eval

    print("=" * 70)
    print("DEMO: drift baseline (snapshot 1 — original responses)")
    print("=" * 70)
    snapshot_holder["n"] = 1
    rc = run_baseline(
        repo / "prompts" / "example.yaml",
        db_path,
        "openai",
        samples=1,
        temperature=0.0,
    )
    print(f"exit code: {rc}\n")

    print("=" * 70)
    print("DEMO: drift run (snapshot 2 — paraphrased / drifted responses)")
    print("=" * 70)
    snapshot_holder["n"] = 2
    rc = run_eval(
        repo / "prompts" / "example.yaml",
        db_path,
        threshold=0.95,
        report_dir=reports,
        provider_name=None,
        samples=None,
        temperature=None,
    )
    print(f"exit code: {rc}\n")

    print("=" * 70)
    print("Generated report (reports/run-2.md):")
    print("=" * 70)
    report_path = reports / "run-2.md"
    print(report_path.read_text(encoding="utf-8"))
    return rc


if __name__ == "__main__":
    sys.exit(main())
