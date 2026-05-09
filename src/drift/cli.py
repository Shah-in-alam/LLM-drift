from pathlib import Path

import typer

from drift.providers import VALID_PROVIDERS
from drift.runner import run_baseline, run_eval

app = typer.Typer(no_args_is_help=True, add_completion=False)

_AUTO_TEMP_FOR_MULTI_SAMPLE = 0.7


@app.callback()
def _main() -> None:
    """LLM Output Drift Detector."""


def _validate_provider(value: str | None) -> str | None:
    if value is not None and value not in VALID_PROVIDERS:
        raise typer.BadParameter(
            f"Unknown provider {value!r}. Choices: {', '.join(VALID_PROVIDERS)}."
        )
    return value


def _validate_samples(value: int) -> int:
    if value < 1:
        raise typer.BadParameter("--samples must be >= 1.")
    return value


def _resolve_temperature(samples: int, temperature: float | None) -> float:
    """If multi-sample and the user didn't pin a temperature, bump to a useful default."""
    if temperature is not None:
        return temperature
    if samples > 1:
        typer.echo(
            f"Note: --samples {samples} > 1 with no --temperature; "
            f"using {_AUTO_TEMP_FOR_MULTI_SAMPLE} so samples are not deterministic."
        )
        return _AUTO_TEMP_FOR_MULTI_SAMPLE
    return 0.0


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
    samples: int = typer.Option(
        1,
        "--samples",
        callback=_validate_samples,
        help="Number of chat samples per prompt (>=1). Use >1 to measure provider noise.",
    ),
    temperature: float = typer.Option(
        None,
        "--temperature",
        help=(
            "Chat temperature. Default 0.0 for single-sample, "
            f"{_AUTO_TEMP_FOR_MULTI_SAMPLE} for multi-sample."
        ),
    ),
) -> None:
    """Run the prompt suite once and store the responses as a baseline."""
    temp = _resolve_temperature(samples, temperature)
    exit_code = run_baseline(prompts, db, provider, samples, temp)
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
    samples: int | None = typer.Option(
        None,
        "--samples",
        help="Samples per prompt. Defaults to whatever the latest baseline used.",
    ),
    temperature: float | None = typer.Option(
        None,
        "--temperature",
        help="Chat temperature. Defaults to whatever the latest baseline used.",
    ),
    psi_threshold: float = typer.Option(
        0.25,
        "--psi-threshold",
        help=(
            "Maximum allowed PSI between baseline and the rolling eval-run window. "
            "0.10 = no shift, 0.10–0.25 = moderate, >0.25 = major shift."
        ),
    ),
    rolling_window: int = typer.Option(
        7,
        "--rolling-window",
        help="Number of recent eval runs (including this one) used for PSI / KL.",
    ),
    edit_threshold: float = typer.Option(
        0.3,
        "--edit-threshold",
        help=(
            "Maximum normalized token-edit distance per prompt before format drift "
            "is flagged. 0.0 = identical text, 1.0 = no token overlap."
        ),
    ),
    p_threshold: float = typer.Option(
        0.05,
        "--p-threshold",
        help=(
            "Mann-Whitney U p-value below which a drift is statistically "
            "significant. Reported alongside cosine; does not gate FAIL on its own."
        ),
    ),
    effect_delta: float = typer.Option(
        0.01,
        "--effect-delta",
        help=(
            "Minimum (mean baseline self-cosine) - (mean baseline-vs-eval cosine) "
            "for a drift to be considered 'significant'. Filters out tiny noise."
        ),
    ),
) -> None:
    """Run an evaluation and compare each prompt against the latest baseline."""
    if samples is not None and samples < 1:
        raise typer.BadParameter("--samples must be >= 1.")
    if rolling_window < 1:
        raise typer.BadParameter("--rolling-window must be >= 1.")
    # If user passed --samples > 1 but no --temperature, auto-bump too.
    resolved_temp = (
        _resolve_temperature(samples, temperature) if samples is not None else temperature
    )
    exit_code = run_eval(
        prompts,
        db,
        threshold,
        report_dir,
        provider,
        samples,
        resolved_temp,
        psi_threshold=psi_threshold,
        rolling_window=rolling_window,
        edit_threshold=edit_threshold,
        p_threshold=p_threshold,
        effect_delta=effect_delta,
    )
    raise typer.Exit(exit_code)


if __name__ == "__main__":
    app()
