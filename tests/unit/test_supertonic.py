"""Tests for the Supertonic engine adapter.

These run without ``supertonic``/``torch`` installed: the engine is constructed
directly and its underlying model is replaced with a fake, so the adapter
logic is exercised without loading any weights.
"""

import numpy as np

from unitts.engines.registry import ENGINE_REGISTRY
from unitts.engines.supertonic_engine import SupertonicEngine


class _FakeSupertonicModel:
    """Stand-in for ``supertonic.TTS``."""

    def __init__(self) -> None:
        self.calls: dict = {}

    def get_voice_style(self, voice_name):
        self.calls["voice_name"] = voice_name
        return f"style-{voice_name}"

    def synthesize(self, text, lang, voice_style, total_steps=8, speed=1.05, **kw):
        self.calls.update(
            {
                "text": text,
                "lang": lang,
                "voice_style": voice_style,
                "total_steps": total_steps,
                "speed": speed,
            }
        )
        return np.zeros((1, 44100), dtype=np.float32), np.array([1.0])


def _engine(**kwargs) -> SupertonicEngine:
    eng = SupertonicEngine(device="cpu", **kwargs)
    eng.model = _FakeSupertonicModel()
    eng._loaded = True
    return eng


def test_registered():
    assert ENGINE_REGISTRY["supertonic"] is SupertonicEngine


def test_metadata():
    eng = SupertonicEngine(device="cpu")
    assert "MIT" in eng.license
    assert eng.default_sample_rate == 44100
    assert not eng.supports_voice_cloning
    assert "tr" in eng.languages
    assert len(eng.languages) == 31


def test_synthesize_defaults():
    eng = _engine()
    result = eng.synthesize("Hello world")
    assert eng.model.calls["voice_name"] == "M1"
    assert eng.model.calls["voice_style"] == "style-M1"
    assert eng.model.calls["lang"] == "en"
    assert eng.model.calls["total_steps"] == 8
    assert result.engine_name == "supertonic"
    assert result.sample_rate == 44100
    assert result.duration_seconds == 1.0
    assert result.audio.shape == (44100,)


def test_synthesize_overrides():
    eng = _engine(voice="F2", lang="tr")
    eng.synthesize("Merhaba", voice="F3", total_steps=12, speed=0.9)
    assert eng.model.calls["voice_name"] == "F3"
    assert eng.model.calls["lang"] == "tr"
    assert eng.model.calls["total_steps"] == 12
    assert eng.model.calls["speed"] == 0.9
