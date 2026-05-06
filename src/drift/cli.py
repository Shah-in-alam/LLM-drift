from pathlib import Path

import typer

from drift.runner import run_baseline, run_eval

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
) -> None:
    """Run an evaluation and compare each prompt against the latest baseline."""
    exit_code = run_eval(prompts, db, threshold, report_dir)
    raise typer.Exit(exit_code)


if __name__ == "__main__":
    app()
