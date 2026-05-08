"""Smoke tests for the scheduling helpers.

We can't actually register a scheduled task or run cron from CI, so these
tests just verify the scripts exist and have the structural pieces a user
following the README would need.
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_cron_example_has_required_shape():
    p = REPO / "scripts" / "cron_example.sh"
    assert p.exists(), f"missing {p}"
    text = p.read_text(encoding="utf-8")

    # First line must be a shebang so cron can execute it directly.
    first_line = text.splitlines()[0]
    assert first_line.startswith("#!"), f"missing shebang: {first_line!r}"

    # Strict mode so a typo in the script doesn't silently swallow failures.
    assert "set -euo pipefail" in text

    # The whole point is to invoke the CLI.
    assert "uv run drift run" in text

    # exec ensures the script's exit code is the CLI's exit code (cron-friendly).
    assert "exec uv run drift run" in text


def test_task_scheduler_ps1_has_required_shape():
    p = REPO / "scripts" / "task_scheduler.ps1"
    assert p.exists(), f"missing {p}"
    text = p.read_text(encoding="utf-8")

    # PowerShell advanced function metadata signals a properly written script.
    assert "[CmdletBinding()]" in text

    # Both lifecycle paths must be present.
    assert "Register-ScheduledTask" in text
    assert "Unregister-ScheduledTask" in text

    # Documented options.
    assert "-IntervalHours" in text
    assert "-Unregister" in text

    # Has to actually run the CLI.
    assert "drift run" in text
