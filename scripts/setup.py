#!/usr/bin/env python3
"""Install one engine into an owned environment; never import or download model weights."""

from __future__ import annotations

import argparse
import json
import platform
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

HIGGS_REVISION = "3d4eb6e49096aa42e7f8f4e72f028823b2d85b73"
MARKER = ".unitts-setup.json"
IMPORTS = {
    "chatterbox": "from chatterbox.tts import ChatterboxTTS",
    "moss-tts": "from transformers import AutoModel, AutoProcessor; import torchcodec",
    "higgs-tts": "from unitts.engines.higgs_tts_engine import HiggsTTSEngine",
    "fish-audio": (
        "from unitts.engines.fish_audio_engine import _ensure_project_root_marker; "
        "_ensure_project_root_marker(); "
        "from fish_speech.inference_engine import TTSInferenceEngine"
    ),
    "qwen3-tts": "from qwen_tts import Qwen3TTSModel",
    "echo-tts": "from echo_tts import EchoTTS; import torchcodec",
    "kokoro": "from kokoro import KPipeline",
    "supertonic": "from supertonic import TTS",
    "neutts": "from neutts import NeuTTS",
}
REMEDIES = {
    "chatterbox": "For missing pkg_resources, reinstall setuptools<82 in this environment.",
    "fish-audio": (
        "For PyAudio build errors, install portaudio19-dev and a C compiler on Debian/Ubuntu."
    ),
    "neutts": "For phonemizer errors, install espeak-ng with your OS package manager.",
    "kokoro": "For phonemizer errors, install espeak-ng with your OS package manager.",
}


def run(command: list[str]) -> None:
    print(f"+ {shlex.join(command)}", flush=True)
    subprocess.run(command, check=True)


def cuda_version() -> tuple[int, int] | None:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None
    result = subprocess.run([executable], capture_output=True, text=True, check=False)
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", result.stdout)
    if result.returncode or not match:
        return None
    return int(match[1]), int(match[2])


def preflight(engine: str, requested_device: str) -> str:
    if engine == "supertonic" and requested_device == "auto":
        return "cpu"
    version = cuda_version() if requested_device != "cpu" else None
    device = requested_device if requested_device != "auto" else ("cuda" if version else "cpu")
    if engine in {"echo-tts", "higgs-tts"} and (platform.system() != "Linux" or device != "cuda"):
        raise RuntimeError(f"{engine} setup requires Linux and an NVIDIA CUDA GPU.")
    if engine == "higgs-tts":
        libc, release = platform.libc_ver()
        if libc != "glibc" or tuple(map(int, release.split(".")[:2])) < (2, 34):
            raise RuntimeError(
                "Higgs needs glibc>=2.34; use Ubuntu 22.04+ or an equivalent Linux image."
            )
    if device == "cuda":
        minimum = (
            (13, 0) if engine == "higgs-tts" else (12, 4) if engine == "chatterbox" else (12, 8)
        )
        if version is None or version < minimum:
            required = ".".join(map(str, minimum))
            raise RuntimeError(
                f"{engine} needs an NVIDIA driver reporting CUDA {required} or newer in "
                "nvidia-smi. Update the driver/expose the GPU, or select --device cpu "
                "for a CPU-capable engine. Installing the CUDA toolkit alone is insufficient."
            )
    if engine in {"moss-tts", "echo-tts", "higgs-tts"} and not shutil.which("ffmpeg"):
        raise RuntimeError(
            "FFmpeg is missing. Install it with your OS package manager "
            "(Debian/Ubuntu: sudo apt-get install ffmpeg; macOS: brew install ffmpeg), "
            "then rerun this command."
        )
    return device


def python_path(environment: Path) -> Path:
    return environment / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")


def identity(root: Path, engine: str, device: str, role: str) -> dict:
    return {"version": 1, "root": str(root), "engine": engine, "device": device, "role": role}


def check_ownership(environment: Path, expected: dict) -> None:
    if not environment.exists() and not environment.is_symlink():
        return
    marker = environment / MARKER
    try:
        owned = not environment.is_symlink() and not marker.is_symlink()
        owned = owned and json.loads(marker.read_text()) == expected
    except (OSError, ValueError):
        owned = False
    if not owned:
        raise RuntimeError(
            f"Refusing to modify {environment}: it is not an environment owned by "
            "this checkout with the same engine/device. Choose the original device, "
            "or move that directory aside before retrying."
        )


def ensure_environment(uv: str, environment: Path, expected: dict) -> Path:
    check_ownership(environment, expected)
    environment.mkdir(exist_ok=True)
    (environment / MARKER).write_text(json.dumps(expected, indent=2) + "\n")
    python = python_path(environment)
    if not python.is_file():
        run([uv, "venv", "--python", "3.12", "--allow-existing", str(environment)])
    return python


def backend(device: str, engine: str) -> list[str]:
    if device == "cuda":
        return ["--torch-backend", "cu124" if engine == "chatterbox" else "cu128"]
    return ["--torch-backend", "cpu"] if platform.system() == "Linux" else []


def packages(engine: str) -> list[str]:
    if engine in {"moss-tts", "echo-tts"}:
        return ["torch==2.9.1", "torchaudio==2.9.1", "torchcodec==0.8.1"]
    if engine == "chatterbox":
        return ["chatterbox-tts==0.1.7", "torch==2.6.0", "torchaudio==2.6.0", "setuptools<82"]
    result = ["torch==2.8.0"]
    if engine in {"fish-audio", "qwen3-tts", "neutts"}:
        result.append("torchaudio==2.8.0")
    if engine == "neutts":
        result.append("setuptools<82")
    if engine == "kokoro":
        result.append("transformers==4.57.3")
    return result


def override_arguments(engine: str) -> list[str]:
    if engine in {"fish-audio", "higgs-tts"}:
        return ["--override", str(Path(__file__).parent / "constraints" / f"{engine}.txt")]
    return []


def check_packages(uv: str, python: Path, engine: str = "") -> None:
    command = [uv, "pip", "check", "--python", str(python)]
    overrides = override_arguments(engine)
    if not overrides:
        run(command)
        return
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    output = result.stdout + result.stderr
    print(output, end="", flush=True)
    if result.returncode:
        expected = (
            r"^The package `descript-audiotools` requires `protobuf>=3\.9\.2,<3\.20`, "
            r"but `[^`]+` is installed$"
        )
        if not (
            result.returncode == 1
            and re.search(r"^Found 1 incompatibility$", output, re.MULTILINE)
            and re.search(expected, output, re.MULTILINE)
        ):
            raise subprocess.CalledProcessError(result.returncode, command)
        print("Using the upstream Protobuf override for descript-audiotools metadata.")
    requirement = next(
        line for line in Path(overrides[1]).read_text().splitlines() if line.startswith("protobuf")
    )
    run(
        [
            str(python),
            "-c",
            (
                "from importlib.metadata import version; "
                "from packaging.specifiers import SpecifierSet; "
                "assert version('protobuf') in "
                f"SpecifierSet({requirement.removeprefix('protobuf')!r}), "
                "'Installed Protobuf does not satisfy the upstream override'"
            ),
        ]
    )


def verify(uv: str, python: Path, imports: str, device: str, engine: str = "") -> None:
    check_packages(uv, python, engine)
    code = "import torch; import unitts; " + imports
    if device == "cuda":
        code += (
            "; assert torch.cuda.is_available(), 'PyTorch cannot access CUDA; check GPU visibility'"
        )
    run([str(python), "-c", code])


def install(engine: str, requested_device: str, root: Path) -> None:
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError(
            "uv is missing. Run: python3 -m pip install uv; then rerun this command."
        )
    root = root.resolve()
    device = preflight(engine, requested_device)
    client_device = "cpu" if engine == "higgs-tts" else device
    environment = root / f".venv-{engine}"
    expected = identity(root, engine, client_device, "engine")
    check_ownership(environment, expected)
    server = root / ".venv-higgs-server"
    server_identity = identity(root, engine, device, "server")
    if engine == "higgs-tts":
        check_ownership(server, server_identity)
    python = ensure_environment(uv, environment, expected)
    requirement = str(root) if engine == "higgs-tts" else f"{root}[{engine}]"
    run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            *backend(client_device, engine),
            "-e",
            requirement,
            *packages(engine),
            *(override_arguments(engine) if engine != "higgs-tts" else []),
        ]
    )
    verify(uv, python, IMPORTS[engine], client_device, engine if engine != "higgs-tts" else "")
    if engine == "higgs-tts":
        server_python = ensure_environment(uv, server, server_identity)
        run(
            [
                uv,
                "pip",
                "install",
                "--python",
                str(server_python),
                "--prerelease=allow",
                "--torch-backend",
                "cu130",
                *override_arguments(engine),
                f"sglang-omni @ git+https://github.com/sgl-project/sglang-omni.git@{HIGGS_REVISION}",
            ]
        )
        check_packages(uv, server_python, engine)
        run(
            [
                str(server_python),
                "-c",
                (
                    "import torch; import torchcodec; import sglang_omni.cli; "
                    "from importlib.resources import files; "
                    "assert files('sglang_omni.models.higgs_tts').joinpath("
                    "'configs/higgs_audio_v2_tokenizer.json').is_file(), "
                    "'Missing Higgs codec config'; "
                    "assert torch.cuda.is_available(), "
                    "'PyTorch cannot access CUDA; check GPU visibility'"
                ),
            ]
        )
    print(
        f"\nInstalled and checked {engine} dependencies ({device}). Weights download on first use."
    )
    if platform.system() == "Windows":
        print(f"Activate: {environment / 'Scripts' / 'Activate.ps1'}")
    else:
        print(f"Activate: source {shlex.quote(str(environment / 'bin' / 'activate'))}")
    if engine == "higgs-tts":
        print("Then start the server: unitts serve --engine higgs-tts")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, choices=sorted(IMPORTS))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args(argv)
    try:
        install(args.engine, args.device, Path(__file__).resolve().parents[1])
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(f"\nSetup failed: {error}", file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError):
            print(
                "Resolve the original error above, then rerun the same setup command.",
                file=sys.stderr,
            )
            if args.engine in {"moss-tts", "echo-tts", "higgs-tts"}:
                print(
                    "TorchCodec errors require compatible system FFmpeg libraries.", file=sys.stderr
                )
            if args.engine in REMEDIES:
                print(REMEDIES[args.engine], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
