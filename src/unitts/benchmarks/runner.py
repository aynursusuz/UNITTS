"""Benchmark runner."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from unitts.datasets import LONG_TEXT_EN
from unitts.engines import ENGINE_REGISTRY, get_engine

console = Console()

# Keep one canonical benchmark text in unitts.datasets so every caller shares it.
DEFAULT_TEXT = LONG_TEXT_EN


def run_benchmark(
    engine_names: list[str] | None = None,
    text: str = DEFAULT_TEXT,
    results_dir: str | Path = "benchmarks/results",
    samples_dir: str | Path = "benchmarks/audio_samples",
    device: str = "auto",
) -> list[dict[str, Any]]:
    """Run benchmarks across the selected engines. Writes one JSON per engine
    under `results_dir/<name>.json` and one WAV under `samples_dir/<name>.wav`.
    """
    results_dir = Path(results_dir)
    samples_dir = Path(samples_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    if engine_names is None:
        engine_names = list(ENGINE_REGISTRY.keys())

    if not engine_names:
        console.print("[red]No engines available. Install engine extras first.[/red]")
        return []

    results: list[dict[str, Any]] = []
    table = Table(title="Benchmark")
    table.add_column("Engine", style="cyan", no_wrap=True)
    table.add_column("Audio (s)", justify="right")
    table.add_column("Inference (s)", justify="right")
    table.add_column("RTF", justify="right")
    table.add_column("VRAM (MB)", justify="right")
    table.add_column("Status", style="bold")

    for name in engine_names:
        try:
            engine = get_engine(name, device=device)

            load_start = time.perf_counter()
            engine.ensure_loaded()
            load_time = time.perf_counter() - load_start
            vram_after_load = engine.get_vram_usage_mb()

            audio_path = samples_dir / f"{name}.wav"
            result = engine.synthesize_to_file(text, audio_path)
            vram_after_inference = engine.get_vram_usage_mb()

            entry = {
                "engine": name,
                "status": "success",
                "text": text,
                "duration_seconds": round(result.duration_seconds, 3),
                "inference_time_seconds": round(result.inference_time_seconds, 3),
                "real_time_factor": round(result.real_time_factor, 4),
                "load_time_seconds": round(load_time, 3),
                "sample_rate": result.sample_rate,
                "vram_after_load_mb": round(vram_after_load, 1) if vram_after_load else None,
                "vram_after_inference_mb": (
                    round(vram_after_inference, 1) if vram_after_inference else None
                ),
                "audio_file": str(audio_path),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            table.add_row(
                name,
                f"{result.duration_seconds:.2f}",
                f"{result.inference_time_seconds:.2f}",
                f"{result.real_time_factor:.3f}",
                f"{vram_after_inference:.0f}" if vram_after_inference else "N/A",
                "[green]OK[/green]",
            )

            engine.unload_model()

        except Exception as e:
            entry = {
                "engine": name,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            table.add_row(name, "-", "-", "-", "-", f"[red]FAIL: {str(e)[:40]}[/red]")
            console.print(f"[red]Error with {name}: {e}[/red]")

        (results_dir / f"{name}.json").write_text(json.dumps(entry, indent=2))
        results.append(entry)

    console.print(table)
    return results
