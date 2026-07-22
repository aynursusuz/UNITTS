"""Tests for the Kokoro engine adapter.

These run without ``kokoro``/``torch`` installed: the engine is constructed
directly and its underlying pipeline is replaced with a fake, so the adapter
logic is exercised without loading any weights.
"""

import numpy as np
import pytest

from unitts.engines.kokoro_engine import KokoroEngine
from unitts.engines.registry import ENGINE_REGISTRY


class _FakePipeline:
    """Stand-in for ``kokoro.KPipeline`` yielding two audio chunks."""

    def __init__(self) -> None:
        self.calls: dict = {}

    def __call__(self, text, voice, speed=1.0, split_pattern=r"\n+", **kw):
        self.calls = {"text": text, "voice": voice, "speed": speed, "split_pattern": split_pattern}
        yield "chunk one", "phonemes", np.zeros(12000, dtype=np.float32)
        yield "chunk two", "phonemes", np.zeros(12000, dtype=np.float32)


def _engine(**kwargs) -> KokoroEngine:
    eng = KokoroEngine(device="cpu", **kwargs)
    eng.model = _FakePipeline()
    eng._loaded = True
    return eng


def test_registered():
    assert ENGINE_REGISTRY["kokoro"] is KokoroEngine


def test_metadata():
    eng = KokoroEngine(device="cpu")
    assert eng.license == "Apache-2.0"
    assert eng.default_sample_rate == 24000
    assert not eng.supports_voice_cloning
    assert "en" in eng.languages


def test_invalid_lang_code():
    with pytest.raises(ValueError, match="lang_code"):
        KokoroEngine(device="cpu", lang_code="x")


def test_synthesize_defaults():
    eng = _engine()
    result = eng.synthesize("Hello world")
    assert eng.model.calls["voice"] == "af_heart"
    assert eng.model.calls["speed"] == 1.0
    assert result.engine_name == "kokoro"
    assert result.sample_rate == 24000
    assert result.duration_seconds == 1.0  # two 12000-sample chunks at 24 kHz
    assert result.metadata["chunks"] == 2


def test_synthesize_overrides():
    eng = _engine(lang_code="b", voice="bf_emma")
    eng.synthesize("Hello", voice="bm_george", speed=1.3)
    assert eng.model.calls["voice"] == "bm_george"
    assert eng.model.calls["speed"] == 1.3


def test_torch_tensor_chunks_are_converted():
    class _FakeTensor:
        def __init__(self, array):
            self._array = array

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._array

    class _TensorPipeline:
        def __call__(self, text, voice, **kw):
            yield "g", "p", _FakeTensor(np.zeros(24000, dtype=np.float32))

    eng = KokoroEngine(device="cpu")
    eng.model = _TensorPipeline()
    eng._loaded = True
    result = eng.synthesize("Hello")
    assert isinstance(result.audio, np.ndarray)
    assert result.duration_seconds == 1.0
