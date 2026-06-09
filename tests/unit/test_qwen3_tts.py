"""Tests for the Qwen3-TTS engine adapter.

These run without ``qwen-tts``/``torch`` installed: the engine class is
constructed directly and its underlying model is replaced with a fake, so the
dispatch logic is exercised without loading any weights.
"""

import numpy as np
import pytest

from unitts.engines.qwen3_tts_engine import Qwen3TTSEngine
from unitts.engines.registry import ENGINE_REGISTRY


class _FakeInner:
    def __init__(self, tts_model_type: str) -> None:
        self.tts_model_type = tts_model_type


class _FakeQwenModel:
    """Stand-in for ``qwen_tts.Qwen3TTSModel`` with a selectable model type."""

    def __init__(self, tts_model_type: str) -> None:
        self.model = _FakeInner(tts_model_type)
        self.calls: dict = {}

    def generate_custom_voice(self, text, speaker, language=None, instruct=None, **kw):
        self.calls = {"fn": "custom_voice", "speaker": speaker, "instruct": instruct}
        return [np.zeros(24000, dtype=np.float32)], 24000

    def generate_voice_design(self, text, instruct, language=None, **kw):
        self.calls = {"fn": "voice_design", "instruct": instruct}
        return [np.zeros(12000, dtype=np.float32)], 24000

    def generate_voice_clone(self, text, language=None, ref_audio=None, ref_text=None, **kw):
        self.calls = {"fn": "voice_clone", "ref_audio": ref_audio, "ref_text": ref_text}
        return [np.zeros(48000, dtype=np.float32)], 24000


def _engine(model_type: str) -> Qwen3TTSEngine:
    eng = Qwen3TTSEngine(device="cpu")
    eng.model = _FakeQwenModel(model_type)
    eng.model_type = model_type
    eng._loaded = True
    return eng


def test_registered():
    assert ENGINE_REGISTRY["qwen3-tts"] is Qwen3TTSEngine


def test_metadata():
    eng = Qwen3TTSEngine(device="cpu")
    assert eng.license == "Apache-2.0"
    assert eng.default_sample_rate == 24000
    assert eng.supports_voice_cloning
    assert eng.supports_emotion_control
    assert "en" in eng.languages
    assert eng.model_path.endswith("CustomVoice")


def test_invalid_dtype():
    with pytest.raises(ValueError, match="dtype"):
        Qwen3TTSEngine(device="cpu", dtype="int8")


def test_custom_voice_dispatch():
    eng = _engine("custom_voice")
    result = eng.synthesize("Hello world", speaker="Aiden", instruct="Very happy.")
    assert eng.model.calls == {"fn": "custom_voice", "speaker": "Aiden", "instruct": "Very happy."}
    assert result.engine_name == "qwen3-tts"
    assert result.sample_rate == 24000
    assert result.duration_seconds == 1.0
    assert result.metadata["mode"] == "custom_voice"


def test_custom_voice_uses_default_speaker():
    eng = _engine("custom_voice")
    eng.synthesize("Hello")
    assert eng.model.calls["speaker"] == "Ryan"


def test_voice_design_requires_instruct():
    eng = _engine("voice_design")
    with pytest.raises(ValueError, match="instruct"):
        eng.synthesize("Hello")
    eng.synthesize("Hello", instruct="A calm narrator")
    assert eng.model.calls["fn"] == "voice_design"


def test_voice_clone_requires_ref_audio():
    eng = _engine("base")
    with pytest.raises(ValueError, match="ref_audio"):
        eng.synthesize("Hello")
    eng.synthesize("Hello", ref_audio="ref.wav", ref_text="reference transcript")
    assert eng.model.calls == {
        "fn": "voice_clone",
        "ref_audio": "ref.wav",
        "ref_text": "reference transcript",
    }
