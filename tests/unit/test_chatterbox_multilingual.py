"""Exercise the multilingual adapter without Chatterbox, CUDA or model weights."""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from unitts.engines import get_engine
from unitts.engines.chatterbox_engine import ChatterboxEngine
from unitts.engines.chatterbox_multilingual_engine import ChatterboxMultilingualEngine


class _FakeWave:
    def __init__(self, samples: int):
        self.samples = samples

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.zeros((1, self.samples), dtype=np.float64)


class _FakeModel:
    sr = 24000

    def __init__(self, samples=24000, events=None):
        self.samples = samples
        self.events = events
        self.calls = []

    def generate(self, text, **kwargs):
        self.calls.append({"text": text, **kwargs})
        if self.events is not None:
            self.events.append("generate")
        return _FakeWave(self.samples)


def _engine(**kwargs):
    engine = ChatterboxMultilingualEngine(device="cpu", **kwargs)
    engine.model = _FakeModel()
    engine._loaded = True
    return engine


def _install_fake_chatterbox(monkeypatch, loader):
    package = ModuleType("chatterbox")
    module = ModuleType("chatterbox.mtl_tts")
    module.ChatterboxMultilingualTTS = loader
    monkeypatch.setitem(sys.modules, "chatterbox", package)
    monkeypatch.setitem(sys.modules, "chatterbox.mtl_tts", module)


def test_registry_keeps_english_and_multilingual_separate():
    engine = get_engine("chatterbox-multilingual", device="cpu")
    assert isinstance(engine, ChatterboxMultilingualEngine)
    assert isinstance(get_engine("chatterbox", device="cpu"), ChatterboxEngine)
    assert ChatterboxEngine.languages == ["en"]
    assert len(engine.languages) == 23
    assert "tr" in engine.languages


def test_pypi_loader_receives_no_checkpoint_parameter(monkeypatch):
    calls = []

    class PyPILoader:
        @classmethod
        def from_pretrained(cls, device):
            calls.append(device)
            model = _FakeModel()
            model.sr = 16000
            return model

    _install_fake_chatterbox(monkeypatch, PyPILoader)
    engine = ChatterboxMultilingualEngine(device="cpu")
    engine.ensure_loaded()
    engine.ensure_loaded()
    assert calls == ["cpu"]
    assert engine.default_sample_rate == 16000


def test_explicit_v3_requires_supported_package(monkeypatch):
    class PyPILoader:
        @classmethod
        def from_pretrained(cls, device):
            raise AssertionError("Unsupported selection must fail before loading weights")

    _install_fake_chatterbox(monkeypatch, PyPILoader)
    engine = ChatterboxMultilingualEngine(device="cpu", t3_model="v3")
    with pytest.raises(ValueError, match="does not support t3_model"):
        engine.ensure_loaded()
    assert engine.model is None
    assert not engine._loaded


def test_explicit_v3_forwarded_to_supported_loader(monkeypatch):
    calls = []

    class UpstreamLoader:
        @classmethod
        def from_pretrained(cls, device, t3_model=None):
            calls.append((device, t3_model))
            return _FakeModel()

    _install_fake_chatterbox(monkeypatch, UpstreamLoader)
    engine = ChatterboxMultilingualEngine(device="cpu", t3_model="v3")
    engine.ensure_loaded()
    assert calls == [("cpu", "v3")]


def test_turkish_and_voice_options_forwarded():
    engine = _engine(language=" TR ")
    result = engine.synthesize(
        "Merhaba dünya.", audio_prompt_path="reference.wav", exaggeration=0.7, cfg_weight=0.3
    )
    assert engine.model.calls == [
        {
            "text": "Merhaba dünya.",
            "language_id": "tr",
            "audio_prompt_path": "reference.wav",
            "exaggeration": 0.7,
            "cfg_weight": 0.3,
        }
    ]
    assert result.metadata["language"] == "tr"
    assert result.metadata["voice_cloned"]
    assert result.metadata["t3_model"] is None
    assert result.audio.shape == (24000,)
    assert result.audio.dtype == np.float32
    assert result.duration_seconds == 1.0
    assert result.engine_name == "chatterbox-multilingual"


def test_language_override_does_not_change_engine_default():
    engine = _engine(language="tr")
    assert engine.synthesize("Hello.", language_id="en").metadata["language"] == "en"
    assert engine.synthesize("Merhaba.").metadata["language"] == "tr"
    assert engine.synthesize("Bonjour.", language="fr").metadata["language"] == "fr"
    assert engine.synthesize("Hallo.", language="DE", language_id="de")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"language": "xx"},
        {"language_id": "auto"},
        {"language": "tr", "language_id": "en"},
    ],
)
def test_invalid_language_rejected_before_loading(kwargs):
    engine = ChatterboxMultilingualEngine(device="cpu")
    with pytest.raises(ValueError):
        engine.synthesize("Merhaba.", **kwargs)
    assert not engine._loaded


def test_blank_text_rejected_before_loading():
    engine = ChatterboxMultilingualEngine(device="cpu")
    with pytest.raises(ValueError, match="text must not be empty"):
        engine.synthesize("  ")
    assert not engine._loaded


@pytest.mark.parametrize("samples", [0, 1])
def test_zero_and_single_sample_output(samples):
    engine = _engine()
    engine.model.samples = samples
    result = engine.synthesize("Hello.")
    assert result.audio.shape == (samples,)
    assert result.duration_seconds == samples / 24000
    if samples == 0:
        assert result.real_time_factor == 0


def test_cuda_timing_waits_for_selected_device(monkeypatch):
    events = []
    ticks = iter([10.0, 10.5])

    def sync(device):
        events.append(("synchronize", device))

    def perf_counter():
        events.append("time")
        return next(ticks)

    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(synchronize=sync))
    )
    monkeypatch.setattr(
        "unitts.engines.chatterbox_multilingual_engine.time.perf_counter", perf_counter
    )
    engine = ChatterboxMultilingualEngine(device="cuda:1", language="tr")
    engine.model = _FakeModel(events=events)
    engine._loaded = True
    result = engine.synthesize("Merhaba.")
    assert events == [
        ("synchronize", "cuda:1"),
        "time",
        "generate",
        ("synchronize", "cuda:1"),
        "time",
    ]
    assert result.inference_time_seconds == 0.5
    assert result.real_time_factor == 0.5


def test_unload_uses_base_lifecycle(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    )
    engine = _engine()
    engine.unload_model()
    assert engine.model is None
    assert not engine._loaded
