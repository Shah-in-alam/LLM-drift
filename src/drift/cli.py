from pathlib import Path

import typer

from drift.providers import VALID_PROVIDERS
from drift.runner import run_baseline, run_eval

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def _main() -> None:
    """LLM Output Drift Detector."""


def _validate_provider(value: str | None) -> str | None:
    if value is not None and value not in VALID_PROVIDERS:
        raise typer.BadParameter(
            f"Unknown provider {value!r}. Choices: {', '.join(VALID_PROVIDERS)}."
        )
    return value


@app.command()
def baseline(
    prompts: Path = typer.Option(
        Path("prompts/example.yaml"),
        "--prompts",
        "-p",
        help="Path to the YAML prompt file.",
    ),
    db: Path = typer.Option(
        Path("drift.db"),
        "--db",
        help="Path to the SQLite database file.",
    ),
    provider: str = typer.Option(
        "openai",
        "--provider",
        callback=_validate_provider,
        help=f"Chat provider. Choices: {', '.join(VALID_PROVIDERS)}.",
    ),
) -> None:
    """Run the prompt suite once and store the responses as a baseline."""
    exit_code = run_baseline(prompts, db, provider)
    raise typer.Exit(exit_code)


@app.command()
def run(
    prompts: Path = typer.Option(
        Path("prompts/example.yaml"),
        "--prompts",
        "-p",
        help="Path to the YAML prompt file.",
    ),
    db: Path = typer.Option(
        Path("drift.db"),
        "--db",
        help="Path to the SQLite database file.",
    ),
    threshold: float = typer.Option(
        0.95,
        "--threshold",
        "-t",
        help="Minimum cosine similarity per prompt before drift is flagged.",
    ),
    report_dir: Path = typer.Option(
        Path("reports"),
        "--report-dir",
        help="Directory to write the markdown drift report into.",
    ),
    provider: str | None = typer.Option(
        None,
        "--provider",
        callback=_validate_provider,
        help=(
            f"Chat provider override. Choices: {', '.join(VALID_PROVIDERS)}. "
            "Defaults to the provider used by the latest baseline."
        ),
    ),
) -> None:
    """Run an evaluation and compare each prompt against the latest baseline."""
    exit_code = run_eval(prompts, db, threshold, report_dir, provider)
    raise typer.Exit(exit_code)


if __name__ == "__main__":
    app()
