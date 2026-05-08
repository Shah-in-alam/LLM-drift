#!/usr/bin/env bash
# Helper for cron: cd into the repo, run `drift run`, and propagate the
# exit code so cron can email on FAIL.
#
# Example crontab line (every 6 hours):
#   0 */6 * * * /path/to/LLM-drift/scripts/cron_example.sh >> /path/to/LLM-drift/logs/drift-run.log 2>&1
#
# Any extra args are passed through to `drift run`, e.g. for a tighter
# threshold the cron line could end with: ... cron_example.sh --threshold 0.97

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p logs

# `uv run` resolves the venv + entry point for us. Exec replaces the shell
# so the exit code from `drift run` is what cron sees.
exec uv run drift run "$@"
