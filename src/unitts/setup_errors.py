"""Actionable context for recognizable engine dependency failures."""

from __future__ import annotations


class EngineSetupError(RuntimeError):
    """An engine cannot load because its runtime needs installation or repair."""


_FFMPEG_HINT = (
    "FFmpeg shared libraries or the matching PyTorch/TorchCodec packages are missing or "
    "incompatible. Install FFmpeg (Debian/Ubuntu: sudo apt-get install ffmpeg; "
    "macOS: brew install ffmpeg), then rerun the engine setup."
)
_PORTAUDIO_HINT = (
    "PyAudio needs PortAudio. Install its system headers (Debian/Ubuntu: "
    "sudo apt-get install portaudio19-dev; macOS: brew install portaudio), "
    "then rerun the engine setup."
)
_ESPEAK_HINT = (
    "The eSpeak phonemizer is unavailable. Install eSpeak NG (Debian/Ubuntu: "
    "sudo apt-get install espeak-ng; macOS: brew install espeak-ng)."
)
_MISSING_IMPORT_HINTS = {
    "pkg_resources": "This dependency needs setuptools<82, which still provides pkg_resources.",
    "torchcodec": _FFMPEG_HINT,
    "pyaudio": _PORTAUDIO_HINT,
}
_LOAD_FAILURE_MARKERS = (
    "cannot open shared object",
    "cannot load",
    "could not load",
    "failed to load",
    "library not loaded",
    "image not found",
    "not found",
    "undefined symbol",
    "symbol not found",
    "not installed",
    "no such file",
)


def engine_setup_error(engine_name: str, error: Exception) -> EngineSetupError | None:
    """Explain dependency failures without reclassifying model or network errors.

    The caller retains the original exception as the cause. Ordinary runtime
    errors, including CUDA OOM, and unrecognized OS errors pass through.
    """
    if isinstance(error, EngineSetupError):
        return None
    message = str(error).lower()
    hint = None
    if isinstance(error, ImportError):
        missing = getattr(error, "name", None)
        root_module = missing.split(".")[0] if missing else ""
        hint = _MISSING_IMPORT_HINTS.get(root_module)
        if hint is None:
            hint = (
                f"Required Python module {missing!r} is missing."
                if isinstance(error, ModuleNotFoundError) and missing
                else "A Python dependency could not be imported; installed versions may conflict."
            )
    elif isinstance(error, AssertionError) and message == "torch not compiled with cuda enabled":
        hint = "This PyTorch build has no CUDA support. Install the engine's CUDA-enabled runtime."
    elif isinstance(error, (OSError, RuntimeError)) and engine_name != "higgs-tts":
        load_failure = any(marker in message for marker in _LOAD_FAILURE_MARKERS)
        if "could not load libtorchcodec" in message or (
            load_failure
            and any(lib in message for lib in ("libavutil", "libavcodec", "libavformat"))
        ):
            hint = _FFMPEG_HINT
        elif load_failure and ("libportaudio" in message or "portaudio library" in message):
            hint = _PORTAUDIO_HINT
        elif load_failure and ("libespeak" in message or message.startswith("espeak ")):
            hint = _ESPEAK_HINT
        elif load_failure and any(
            lib in message for lib in ("libtorch", "libc10", "libtorchaudio")
        ):
            hint = (
                "PyTorch and its native extensions are incompatible; "
                "reinstall this engine's runtime."
            )
        elif any(
            marker in message
            for marker in (
                "found no nvidia driver",
                "no cuda gpus are available",
                "cuda driver version is insufficient",
                "torch not compiled with cuda enabled",
            )
        ):
            hint = (
                "CUDA is unavailable. Check nvidia-smi and the NVIDIA driver, and install "
                "the engine's CUDA-enabled PyTorch runtime."
            )
        elif isinstance(error, FileNotFoundError) and error.filename in (
            "ffmpeg",
            "espeak",
            "espeak-ng",
        ):
            hint = _FFMPEG_HINT if error.filename == "ffmpeg" else _ESPEAK_HINT
    if hint is None:
        return None
    return EngineSetupError(
        f"Cannot load {engine_name!r}: {hint}\n"
        f"From the UNITTS checkout, run: python3 scripts/setup.py --engine {engine_name}\n"
        f"Use the environment created for this engine. Original error: {error}"
    )
