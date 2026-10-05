"""Run short, reproducible UNITTS checks on one CUDA GPU.

Usage: python gpu_smoke_test.py --engine moss-tts --output-dir smoke/moss
Run MOSS-TTS from its own environment, or start the Higgs server separately.
Higgs timing includes the HTTP round trip; client PyTorch cannot measure the
server's VRAM, so no VRAM comparison is reported. This checks inference and audio
integrity; it does not score pronunciation or voice similarity.
"""

import argparse
import importlib.metadata
import json
import platform
import time
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from unitts.engines import get_engine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["moss-tts", "higgs-tts"], default="moss-tts")
    parser.add_argument("--model-path")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to run this GPU test on CPU")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(2026)
    np.random.seed(2026)
    options = {"device": "cuda"}
    if args.model_path:
        options["model_path"] = args.model_path
    is_local = args.engine == "moss-tts"
    if not is_local:
        options["base_url"] = args.base_url
    engine = get_engine(args.engine, **options)
    report = {
        "engine": args.engine,
        "options": options,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_mb": torch.cuda.get_device_properties(0).total_memory / 1024**2,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "packages": {},
        "cases": [],
        "note": "First load can include downloads. Warm samples, not a quality ranking.",
    }
    for package in ["transformers", "torchaudio", "torchcodec"]:
        with suppress(importlib.metadata.PackageNotFoundError):
            report["packages"][package] = importlib.metadata.version(package)
    generation_options = {"max_new_tokens": 512}
    print(f"Loading {args.engine} on {report['gpu']}", flush=True)
    try:
        start = time.perf_counter()
        engine.ensure_loaded()
        torch.cuda.synchronize()
        load_key = "first_load_seconds_including_downloads" if is_local else "server_check_seconds"
        report[load_key] = time.perf_counter() - start
        print("Model loaded; warming up", flush=True)
        warmup_options = {"language": "en"} if is_local else {"seed": 2026}
        engine.synthesize("Hello, this is a short warmup.", **warmup_options, **generation_options)
        cases = [
            ("tr", "Merhaba! Bugün İstanbul'da hava çok güzel. Yeni ses modelimizi deniyoruz."),
            (
                "en",
                "Hello! The weather in Istanbul is beautiful today. "
                "We are testing our new voice model.",
            ),
            (
                "fr",
                "Bonjour ! Il fait très beau à Istanbul aujourd'hui. "
                "Nous testons notre nouveau modèle vocal.",
            ),
            ("tr-clone", "Bu cümle, ilk örnekte üretilen ses referans alınarak okunuyor."),
        ]
        for label, text in cases:
            lang = label.split("-")[0]
            kwargs = dict(generation_options)
            if is_local:
                kwargs["language"] = lang
            else:
                kwargs["seed"] = 2026
            if label == "tr-clone":
                kwargs["ref_audio"] = str(args.output_dir / "tr.wav")
                if not is_local:
                    kwargs["ref_text"] = cases[0][1]
            print(f"Synthesizing {label}", flush=True)
            torch.cuda.reset_peak_memory_stats()
            result = engine.synthesize_to_file(text, args.output_dir / f"{label}.wav", **kwargs)
            audio = result.audio
            if audio.size == 0 or not np.isfinite(audio).all() or not np.any(audio):
                raise RuntimeError(f"Invalid, silent or empty audio in {label}")
            item = {
                "case": label,
                "text": text,
                "language": lang,
                "sample_rate": result.sample_rate,
                "shape": list(audio.shape),
                "duration_seconds": result.duration_seconds,
                "inference_time_seconds": result.inference_time_seconds,
                "real_time_factor": result.real_time_factor,
                "peak_allocated_vram_mb": (
                    torch.cuda.max_memory_allocated() / 1024**2 if is_local else None
                ),
                "peak_reserved_vram_mb": (
                    torch.cuda.max_memory_reserved() / 1024**2 if is_local else None
                ),
                "rms": float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))),
                "metadata": result.metadata,
            }
            report["cases"].append(item)
            print(json.dumps(item, ensure_ascii=False), flush=True)
        report["status"] = "success"
    except Exception as error:
        report["status"] = "error"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        engine.unload_model()
        report["allocated_vram_mb_after_unload"] = (
            torch.cuda.memory_allocated() / 1024**2 if is_local else None
        )
        if not is_local:
            report["server_lifecycle"] = "Managed separately; adapter unload leaves server running"
        (args.output_dir / "result.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Results saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
