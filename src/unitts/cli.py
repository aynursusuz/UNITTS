"""CLI for unitts."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="unitts",
    help="Run and compare open-source TTS models.",
    add_completion=False,
)
console = Console()


@app.command()
def serve(
    engine: Annotated[str, typer.Option("--engine", "-e", help="Server engine name")],
    host: Annotated[str, typer.Option(help="Listen address")] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535, help="Listen port")] = 8000,
    server_python: Annotated[
        Path | None, typer.Option(help="Python in an existing SGLang-Omni environment")
    ] = None,
) -> None:
    """Start the Higgs server in this terminal; stop it with Ctrl+C."""
    from unitts.server import serve_higgs
    from unitts.setup_errors import EngineSetupError

    if engine != "higgs-tts":
        raise typer.BadParameter("Only higgs-tts needs a separate server.", param_hint="--engine")
    try:
        serve_higgs(host=host, port=port, server_python=server_python)
    except (EngineSetupError, OSError) as exc:
        console.print(f"Setup error: {exc}", style="red", markup=False)
        raise typer.Exit(code=1) from exc


@app.command()
def list_engines() -> None:
    """List all available TTS engines."""
    from unitts.engines import list_engines as _list

    engines = _list()
    table = Table(title="Available TTS Engines")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_column("Languages", style="green")
    table.add_column("Clone", style="yellow")
    table.add_column("Stream", style="yellow")
    table.add_column("License", style="dim")

    for e in engines:
        table.add_row(
            e["name"],
            e["description"],
            ", ".join(e["languages"][:3]) + ("..." if len(e["languages"]) > 3 else ""),
            "Yes" if e["voice_cloning"] else "No",
            "Yes" if e["streaming"] else "No",
            e["license"],
        )
    console.print(table)


@app.command()
def synthesize(
    text: Annotated[str, typer.Argument(help="Text to synthesize")],
    engine: Annotated[str, typer.Option("--engine", "-e", help="TTS engine name")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output WAV file")] = Path(
        "output.wav"
    ),
    device: Annotated[str, typer.Option("--device", "-d", help="Device: auto, cuda, cpu")] = "auto",
) -> None:
    """Synthesize speech from text with the given engine."""
    from unitts.engines import get_engine
    from unitts.setup_errors import EngineSetupError

    console.print(f"[cyan]Loading engine:[/cyan] {engine}")
    tts = None
    try:
        tts = get_engine(engine, device=device)
        tts.ensure_loaded()
        console.print(f"[cyan]Synthesizing:[/cyan] {text[:80]}...")
        result = tts.synthesize_to_file(text, output)
    except EngineSetupError as exc:
        console.print(f"Setup error: {exc}", style="red", markup=False)
        raise typer.Exit(code=1) from exc
    finally:
        if tts is not None:
            try:
                tts.unload_model()
            except Exception as exc:
                console.print(f"Could not unload {engine}: {exc}", style="yellow", markup=False)

    console.print(f"[green]Done![/green] Saved to {output}")
    console.print(f"  Duration: {result.duration_seconds:.2f}s")
    console.print(f"  Inference: {result.inference_time_seconds:.2f}s")
    console.print(f"  RTF: {result.real_time_factor:.3f}x")


@app.command()
def benchmark(
    engines: Annotated[
        list[str] | None,
        typer.Option("--engine", "-e", help="Engines to benchmark (repeatable)"),
    ] = None,
    text: Annotated[str | None, typer.Option("--text", "-t", help="Text to benchmark with")] = None,
    results_dir: Annotated[Path, typer.Option("--results-dir")] = Path("benchmarks/results"),
    samples_dir: Annotated[Path, typer.Option("--samples-dir")] = Path("benchmarks/audio_samples"),
    device: Annotated[str, typer.Option("--device", "-d")] = "auto",
) -> None:
    """Run benchmarks across the selected TTS engines."""
    from unitts.benchmarks.runner import DEFAULT_TEXT, run_benchmark

    run_benchmark(
        engine_names=engines,
        text=text or DEFAULT_TEXT,
        results_dir=results_dir,
        samples_dir=samples_dir,
        device=device,
    )


if __name__ == "__main__":
    app()
