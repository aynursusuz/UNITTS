"""Tests for the Echo-TTS engine adapter.

These run without ``echo-tts``/``torch`` installed: the engine is constructed
directly and its underlying model is replaced with a fake, so the adapter logic
is exercised without loading any weights.
"""

import numpy as np
import pytest

from unitts.engines.echo_tts_engine import EchoTTSEngine
from unitts.engines.registry import ENGINE_REGISTRY


class _FakeTensor:
    """Minimal stand-in exposing the tensor methods the adapter calls."""

    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._array


class _FakeEchoModel:
    def __init__(self) -> None:
        self.calls: dict = {}

    def synthesize(self, text, speaker_audio=None, seed=0, num_steps=40, sequence_length=640, **kw):
        self.calls = {
            "text": text,
            "speaker_audio": speaker_audio,
            "seed": seed,
            "num_steps": num_steps,
            "sequence_length": sequence_length,
        }
        return _FakeTensor(np.zeros(44100, dtype=np.float32)), 44100


def _engine() -> EchoTTSEngine:
    eng = EchoTTSEngine(device="cpu")
    eng.model = _FakeEchoModel()
    eng._loaded = True
    return eng


def test_registered():
    assert ENGINE_REGISTRY["echo-tts"] is EchoTTSEngine


def test_metadata():
    eng = EchoTTSEngine(device="cpu")
    assert eng.supports_voice_cloning
    assert eng.languages == ["en"]
    assert "MIT" in eng.license


def test_invalid_dtype():
    with pytest.raises(ValueError, match="dtype"):
        EchoTTSEngine(device="cpu", dtype="int8")


def test_synthesize_defaults():
    eng = _engine()
    result = eng.synthesize("Hello from Echo.")
    assert eng.model.calls["text"] == "Hello from Echo."
    assert eng.model.calls["num_steps"] == 40
    assert eng.model.calls["sequence_length"] == 640
    assert result.sample_rate == 44100
    assert result.duration_seconds == 1.0
    assert result.engine_name == "echo-tts"
    assert result.metadata["voice_cloned"] is False


def test_synthesize_voice_clone_and_overrides():
    eng = _engine()
    eng.synthesize("Hi", speaker_audio="ref.wav", num_steps=16, sequence_length=256)
    assert eng.model.calls["speaker_audio"] == "ref.wav"
    assert eng.model.calls["num_steps"] == 16
    assert eng.model.calls["sequence_length"] == 256
