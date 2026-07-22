"""Tests for the F5-TTS engine adapter.

These run without ``f5-tts``/``torch`` installed: the engine is constructed
directly and its underlying model is replaced with a fake, so the adapter
logic is exercised without loading any weights.
"""

import numpy as np

from unitts.engines.f5_tts_engine import F5TTSEngine
from unitts.engines.registry import ENGINE_REGISTRY


class _FakeF5Model:
    """Stand-in for ``f5_tts.api.F5TTS``."""

    def __init__(self) -> None:
        self.calls: dict = {}
        self.seed = 42

    def infer(self, ref_file, ref_text, gen_text, nfe_step=32, speed=1.0, seed=None, **kw):
        self.calls = {
            "ref_file": ref_file,
            "ref_text": ref_text,
            "gen_text": gen_text,
            "nfe_step": nfe_step,
            "speed": speed,
            "seed": seed,
        }
        return np.zeros(24000, dtype=np.float32), 24000, None


def _engine(**kwargs) -> F5TTSEngine:
    eng = F5TTSEngine(device="cpu", **kwargs)
    eng.model = _FakeF5Model()
    eng._loaded = True
    return eng


def test_registered():
    assert ENGINE_REGISTRY["f5-tts"] is F5TTSEngine


def test_metadata():
    eng = F5TTSEngine(device="cpu")
    assert eng.supports_voice_cloning
    assert "MIT" in eng.license
    assert eng.default_sample_rate == 24000
    assert eng.model_name == "F5TTS_v1_Base"


def test_falls_back_to_bundled_reference(monkeypatch):
    eng = _engine()
    monkeypatch.setattr(
        F5TTSEngine, "_bundled_reference", staticmethod(lambda: ("bundled.wav", "bundled text"))
    )
    eng.synthesize("Hello")
    assert eng.model.calls["ref_file"] == "bundled.wav"
    assert eng.model.calls["ref_text"] == "bundled text"


def test_explicit_ref_text_survives_bundled_fallback(monkeypatch):
    eng = _engine()
    monkeypatch.setattr(
        F5TTSEngine, "_bundled_reference", staticmethod(lambda: ("bundled.wav", "bundled text"))
    )
    eng.synthesize("Hello", ref_text="my transcript")
    assert eng.model.calls["ref_text"] == "my transcript"


def test_synthesize_defaults():
    eng = _engine()
    result = eng.synthesize("Hello world", ref_audio="ref.wav", ref_text="reference transcript")
    assert eng.model.calls["ref_file"] == "ref.wav"
    assert eng.model.calls["ref_text"] == "reference transcript"
    assert eng.model.calls["gen_text"] == "Hello world"
    assert eng.model.calls["nfe_step"] == 32
    assert result.engine_name == "f5-tts"
    assert result.sample_rate == 24000
    assert result.duration_seconds == 1.0
    assert result.metadata["seed"] == 42


def test_empty_ref_text_triggers_auto_transcription():
    eng = _engine()
    eng.synthesize("Hello", ref_audio="ref.wav")
    assert eng.model.calls["ref_text"] == ""


def test_synthesize_overrides():
    eng = _engine(nfe_step=16, speed=0.9)
    eng.synthesize("Hi", ref_audio="ref.wav", nfe_step=64, speed=1.2, seed=7)
    assert eng.model.calls["nfe_step"] == 64
    assert eng.model.calls["speed"] == 1.2
    assert eng.model.calls["seed"] == 7
