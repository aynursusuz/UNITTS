"""Tests for the NeuTTS engine adapter.

These run without ``neutts``/``torch`` installed: the engine is constructed
directly and its underlying model is replaced with a fake, so the adapter
logic is exercised without loading any weights.
"""

import numpy as np
import pytest

from unitts.engines.neutts_engine import NeuTTSEngine
from unitts.engines.registry import ENGINE_REGISTRY


class _FakeNeuTTSModel:
    """Stand-in for ``neutts.NeuTTS``."""

    def __init__(self) -> None:
        self.calls: dict = {}

    def encode_reference(self, ref_audio):
        self.calls["ref_audio"] = ref_audio
        return "codes"

    def infer(self, text, ref_codes, ref_text, **kw):
        self.calls.update({"text": text, "ref_codes": ref_codes, "ref_text": ref_text})
        return np.zeros(24000, dtype=np.float32)


def _engine(**kwargs) -> NeuTTSEngine:
    eng = NeuTTSEngine(device="cpu", **kwargs)
    eng.model = _FakeNeuTTSModel()
    eng._loaded = True
    return eng


def test_registered():
    assert ENGINE_REGISTRY["neutts"] is NeuTTSEngine


def test_metadata():
    eng = NeuTTSEngine(device="cpu")
    assert eng.supports_voice_cloning
    assert eng.default_sample_rate == 24000
    assert eng.backbone_repo == "neuphonic/neutts-air"
    assert "Apache" in eng.license


def test_requires_reference():
    eng = _engine()
    with pytest.raises(ValueError, match="ref_audio"):
        eng.synthesize("Hello")
    with pytest.raises(ValueError, match="ref_text"):
        eng.synthesize("Hello", ref_audio="ref.wav")


def test_synthesize():
    eng = _engine()
    result = eng.synthesize("Hello world", ref_audio="ref.wav", ref_text="reference transcript")
    assert eng.model.calls == {
        "ref_audio": "ref.wav",
        "ref_codes": "codes",
        "text": "Hello world",
        "ref_text": "reference transcript",
    }
    assert result.engine_name == "neutts"
    assert result.sample_rate == 24000
    assert result.duration_seconds == 1.0
    assert result.metadata["backbone_repo"] == "neuphonic/neutts-air"


def test_custom_backbone():
    eng = _engine(backbone_repo="neuphonic/neutts-nano")
    eng.synthesize("Hi", ref_audio="ref.wav", ref_text="hi")
    assert eng.backbone_repo == "neuphonic/neutts-nano"
