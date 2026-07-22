"""Tests for the VoxCPM engine adapter.

These run without ``voxcpm``/``torch`` installed: the engine is constructed
directly and its underlying model is replaced with a fake, so the adapter
logic is exercised without loading any weights.
"""

import numpy as np

from unitts.engines.registry import ENGINE_REGISTRY
from unitts.engines.voxcpm_engine import VoxCPMEngine


class _FakeVoxCPMModel:
    """Stand-in for ``voxcpm.VoxCPM``."""

    def __init__(self) -> None:
        self.calls: dict = {}

    def generate(self, text, **kwargs):
        self.calls = {"text": text, **kwargs}
        return np.zeros(48000, dtype=np.float32)


def _engine(**kwargs) -> VoxCPMEngine:
    eng = VoxCPMEngine(device="cpu", **kwargs)
    eng.model = _FakeVoxCPMModel()
    eng._loaded = True
    return eng


def test_registered():
    assert ENGINE_REGISTRY["voxcpm"] is VoxCPMEngine


def test_metadata():
    eng = VoxCPMEngine(device="cpu")
    assert eng.license == "Apache-2.0"
    assert eng.default_sample_rate == 48000
    assert eng.supports_voice_cloning
    assert eng.supports_emotion_control
    assert "tr" in eng.languages
    assert len(eng.languages) == 30
    assert eng.model_path == "openbmb/VoxCPM2"


def test_synthesize_defaults():
    eng = _engine()
    result = eng.synthesize("Hello world")
    assert eng.model.calls["text"] == "Hello world"
    assert eng.model.calls["cfg_value"] == 2.0
    assert eng.model.calls["inference_timesteps"] == 10
    assert "reference_wav_path" not in eng.model.calls
    assert result.engine_name == "voxcpm"
    assert result.sample_rate == 48000
    assert result.duration_seconds == 1.0
    assert result.metadata["voice_cloned"] is False


def test_synthesize_voice_clone():
    eng = _engine()
    result = eng.synthesize("Hi", ref_audio="ref.wav", seed=7)
    assert eng.model.calls["reference_wav_path"] == "ref.wav"
    assert eng.model.calls["seed"] == 7
    assert result.metadata["voice_cloned"] is True


def test_synthesize_prompt_clone_and_overrides():
    eng = _engine(cfg_value=1.5, inference_timesteps=20)
    eng.synthesize(
        "Hi",
        prompt_audio="prompt.wav",
        prompt_text="prompt transcript",
        inference_timesteps=32,
    )
    assert eng.model.calls["prompt_wav_path"] == "prompt.wav"
    assert eng.model.calls["prompt_text"] == "prompt transcript"
    assert eng.model.calls["cfg_value"] == 1.5
    assert eng.model.calls["inference_timesteps"] == 32
