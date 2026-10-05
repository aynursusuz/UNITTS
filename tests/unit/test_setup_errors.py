"""Dependency diagnostics must help setup without masking inference failures."""

import pytest

from unitts.engines.base import TTSEngine
from unitts.setup_errors import EngineSetupError, engine_setup_error


class _Engine(TTSEngine):
    name = "moss-tts"

    def __init__(self, error):
        super().__init__(device="cpu")
        self.error = error
        self.loads = 0

    def load_model(self):
        self.loads += 1
        if self.error is not None:
            raise self.error
        self.model = object()

    def synthesize(self, text, **kwargs):
        self.ensure_loaded()


@pytest.mark.parametrize(
    ("error", "hint"),
    [
        (
            ModuleNotFoundError("No module named 'transformers'", name="transformers"),
            "transformers",
        ),
        (ModuleNotFoundError("No module named 'einops'", name="einops"), "einops"),
        (
            ModuleNotFoundError("No module named 'pkg_resources'", name="pkg_resources"),
            "setuptools<82",
        ),
        (ModuleNotFoundError("No module named 'torchcodec'", name="torchcodec"), "FFmpeg"),
        (ModuleNotFoundError("No module named 'pyaudio'", name="pyaudio"), "portaudio19-dev"),
        (ImportError("cannot import name 'AutoModel' from transformers"), "versions may conflict"),
        (
            RuntimeError("Could not load libtorchcodec. Likely causes: FFmpeg not installed"),
            "FFmpeg",
        ),
        (OSError("libavutil.so.59: cannot open shared object file"), "FFmpeg"),
        (OSError("libportaudio.so.2: cannot open shared object file"), "portaudio19-dev"),
        (RuntimeError("espeak not installed on your system"), "espeak-ng"),
        (OSError("libtorchaudio.so: undefined symbol: _ZN5torch"), "native extensions"),
        (RuntimeError("Found no NVIDIA driver on your system"), "nvidia-smi"),
        (AssertionError("Torch not compiled with CUDA enabled"), "no CUDA support"),
        (FileNotFoundError(2, "No such file or directory", "ffmpeg"), "FFmpeg"),
    ],
)
def test_load_error_has_remedy_and_preserves_cause(error, hint):
    engine = _Engine(error)
    with pytest.raises(EngineSetupError) as caught:
        engine.ensure_loaded()
    assert hint in str(caught.value)
    assert "python3 scripts/setup.py --engine moss-tts" in str(caught.value)
    assert str(error) in str(caught.value)
    assert caught.value.__cause__ is error
    assert not engine._loaded


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"),
        RuntimeError("Unexpected model output shape"),
        OSError("401 Unauthorized when downloading model weights"),
        OSError("Cannot load tokenizer for missing/model"),
        OSError("Cannot load tokenizer for espeak/model"),
        FileNotFoundError(2, "No such file or directory", "model.safetensors"),
        ValueError("Unsupported dtype"),
        AssertionError("Model tensor shape did not match"),
        EngineSetupError("Already diagnosed"),
    ],
)
def test_unrelated_errors_are_not_reclassified(error):
    engine = _Engine(error)
    with pytest.raises(type(error)) as caught:
        engine.ensure_loaded()
    assert caught.value is error
    assert not engine._loaded


def test_failed_load_can_be_retried_after_repair():
    engine = _Engine(ModuleNotFoundError("Missing transformers", name="transformers"))
    with pytest.raises(EngineSetupError):
        engine.ensure_loaded()
    engine.error = None
    engine.ensure_loaded()
    engine.ensure_loaded()
    assert engine._loaded
    assert engine.loads == 2


def test_remote_server_errors_do_not_suggest_installing_local_libraries():
    error = RuntimeError("Higgs TTS server returned HTTP 500: Could not load libtorchcodec")
    assert engine_setup_error("higgs-tts", error) is None
