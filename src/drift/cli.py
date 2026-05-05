from pathlib import Path

import typer

from drift.runner import run_baseline

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def _main() -> None:
    """LLM Output Drift Detector."""


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
) -> None:
    """Run the prompt suite once and store the responses as a baseline."""
    exit_code = run_baseline(prompts, db)
    raise typer.Exit(exit_code)


if __name__ == "__main__":
    app()
