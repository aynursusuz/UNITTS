"""MOSS-TTS adapter contracts, with fake upstream models and no downloaded weights."""

import gc
import sys
import weakref
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf
import torch

from unitts.engines import get_engine
from unitts.engines import moss_tts_engine as moss
from unitts.engines.moss_tts_engine import MossTTSEngine
from unitts.engines.registry import ENGINE_REGISTRY


def _upstream(waveform=None, sample_rate=24000):
    if waveform is None:
        waveform = torch.zeros(1, sample_rate)
    codec = Mock()
    codec.to.return_value = codec
    processor = Mock(
        audio_tokenizer=codec,
        model_config=SimpleNamespace(sampling_rate=sample_rate),
    )
    processor.return_value = {"input_ids": Mock(), "attention_mask": Mock()}
    processor.decode.return_value = [SimpleNamespace(audio_codes_list=[waveform])]
    model = Mock()
    model.to.return_value = model
    return processor, model


def _engine(waveform=None, sample_rate=24000, **kwargs):
    engine = MossTTSEngine(device=kwargs.pop("device", "cpu"), **kwargs)
    engine.processor, engine.model = _upstream(waveform, sample_rate)
    engine.default_sample_rate = sample_rate
    engine._loaded = True
    return engine


def test_registry_and_metadata(monkeypatch):
    monkeypatch.delenv("MOSS_TTS_MODEL", raising=False)
    engine = get_engine("moss-tts", device="cpu")
    assert ENGINE_REGISTRY["moss-tts"] is MossTTSEngine
    assert isinstance(engine, MossTTSEngine)
    assert engine.model_path == "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
    assert engine.license == "Apache-2.0"
    assert len(engine.languages) == 31
    assert "tr" in engine.languages
    assert engine.supports_voice_cloning
    assert not engine.supports_streaming
    assert engine.model is None
    assert engine.processor is None
    assert not engine._loaded


def test_checkpoint_environment_and_explicit_override(monkeypatch):
    monkeypatch.setenv("MOSS_TTS_MODEL", "local/environment-model")
    assert MossTTSEngine(device="cpu").model_path == "local/environment-model"
    assert MossTTSEngine(device="cpu", model_path="explicit").model_path == "explicit"


@pytest.mark.parametrize("dtype", ["int8", "half", "BF16"])
def test_invalid_dtype(dtype):
    with pytest.raises(ValueError, match="dtype"):
        MossTTSEngine(device="cpu", dtype=dtype)


@pytest.mark.parametrize(
    ("language", "expected"),
    [(None, None), ("auto", None), ("AUTO", None), ("tr", "Turkish"), ("turkish", "Turkish")],
)
def test_language_normalization(language, expected):
    assert MossTTSEngine(device="cpu", language=language).language == expected


def test_invalid_language_rejected_before_loading():
    with pytest.raises(ValueError, match="Unsupported MOSS-TTS language"):
        MossTTSEngine(device="cpu", language="xx")
    engine = MossTTSEngine(device="cpu")
    engine.load_model = Mock()
    with pytest.raises(ValueError, match="Unsupported MOSS-TTS language"):
        engine.synthesize("Merhaba", language="xx")
    engine.load_model.assert_not_called()


@pytest.mark.parametrize(
    ("device", "dtype", "attention", "expected_dtype", "expected_attention"),
    [
        ("cpu", "auto", None, torch.float32, "eager"),
        ("cuda:1", "auto", None, torch.bfloat16, "sdpa"),
        ("cuda", "float16", "flash_attention_2", torch.float16, "flash_attention_2"),
    ],
)
def test_lazy_load_places_both_models_and_resolves_dtype(
    monkeypatch, device, dtype, attention, expected_dtype, expected_attention
):
    processor, model = _upstream(sample_rate=48000)
    auto_processor = Mock()
    auto_processor.from_pretrained.return_value = processor
    auto_model = Mock()
    auto_model.from_pretrained.return_value = model
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoProcessor=auto_processor, AutoModel=auto_model),
    )

    engine = MossTTSEngine(
        device=device, model_path="test/checkpoint", dtype=dtype, attn_implementation=attention
    )
    auto_processor.from_pretrained.assert_not_called()
    auto_model.from_pretrained.assert_not_called()
    engine.ensure_loaded()
    engine.ensure_loaded()

    auto_processor.from_pretrained.assert_called_once_with(
        "test/checkpoint", trust_remote_code=True
    )
    auto_model.from_pretrained.assert_called_once_with(
        "test/checkpoint",
        trust_remote_code=True,
        dtype=expected_dtype,
        attn_implementation=expected_attention,
    )
    processor.audio_tokenizer.to.assert_called_once_with(device)
    processor.audio_tokenizer.eval.assert_called_once_with()
    model.to.assert_called_once_with(device)
    model.eval.assert_called_once_with()
    assert engine.processor is processor
    assert engine.model is model
    assert engine.default_sample_rate == 48000
    assert engine._loaded


def test_generation_uses_one_conversation_and_forwards_reference_and_options(tmp_path):
    engine = _engine(language="en")
    reference = tmp_path / "reference.wav"
    result = engine.synthesize(
        "Merhaba dünya.",
        language="tr",
        ref_audio=reference,
        tokens=75,
        max_new_tokens=300,
        audio_temperature=0.8,
    )

    engine.processor.build_user_message.assert_called_once_with(
        text="Merhaba dünya.", language="Turkish", reference=[str(reference)], tokens=75
    )
    message = engine.processor.build_user_message.return_value
    engine.processor.assert_called_once_with([[message]], mode="generation")
    batch = engine.processor.return_value
    batch["input_ids"].to.assert_called_once_with("cpu")
    batch["attention_mask"].to.assert_called_once_with("cpu")
    engine.model.generate.assert_called_once_with(
        input_ids=batch["input_ids"].to.return_value,
        attention_mask=batch["attention_mask"].to.return_value,
        max_new_tokens=300,
        do_sample=True,
        audio_temperature=0.8,
        audio_top_p=0.8,
        audio_top_k=25,
        audio_repetition_penalty=1.0,
    )
    engine.processor.decode.assert_called_once_with(engine.model.generate.return_value)
    assert result.metadata["language"] == "Turkish"
    assert result.metadata["voice_cloning"] is True
    assert result.text == "Merhaba dünya."


def test_default_generation_options_and_language_override():
    engine = _engine(language="tr")
    engine.synthesize("Merhaba")
    assert engine.processor.build_user_message.call_args.kwargs == {
        "text": "Merhaba",
        "language": "Turkish",
        "reference": None,
        "tokens": None,
    }
    assert engine.model.generate.call_args.kwargs["max_new_tokens"] == 4096
    assert engine.model.generate.call_args.kwargs["audio_temperature"] == 1.7
    result = engine.synthesize("Hello", language="auto")
    assert engine.processor.build_user_message.call_args.kwargs["language"] is None
    assert result.metadata["voice_cloning"] is False


@pytest.mark.parametrize("shape", [(24000,), (1, 24000), (2, 24000)])
def test_audio_layout_duration_and_dtype(shape):
    waveform = torch.linspace(-0.5, 0.5, 24000, dtype=torch.float64)
    if len(shape) == 2:
        waveform = torch.stack([waveform] * shape[0])
    engine = _engine(waveform)
    result = engine.synthesize("Hello")

    expected = waveform.numpy()
    if len(shape) == 2:
        expected = expected[0] if shape[0] == 1 else expected.T
    np.testing.assert_allclose(result.audio, expected.astype(np.float32))
    assert result.audio.dtype == np.float32
    assert result.audio.flags.c_contiguous
    assert result.duration_seconds == 1.0
    assert result.sample_rate == 24000
    assert result.engine_name == "moss-tts"
    assert result.metadata["channels"] == (2 if shape[0] == 2 else 1)
    assert result.real_time_factor == result.inference_time_seconds


def test_stereo_wav_preserves_frames_and_channels(tmp_path):
    waveform = torch.stack([torch.full((24000,), 0.25), torch.full((24000,), -0.5)])
    engine = _engine(waveform)
    output = tmp_path / "nested" / "stereo.wav"
    result = engine.synthesize_to_file("Hello", output)

    audio, sample_rate = sf.read(output, dtype="float32")
    assert sample_rate == result.sample_rate == 24000
    assert audio.shape == (24000, 2)
    np.testing.assert_array_equal(audio, result.audio)


@pytest.mark.parametrize("text", ["", " \n\t"])
def test_empty_text_rejected_before_loading(text):
    engine = MossTTSEngine(device="cpu")
    engine.load_model = Mock()
    with pytest.raises(ValueError, match="text must not be empty"):
        engine.synthesize(text)
    engine.load_model.assert_not_called()


@pytest.mark.parametrize("tokens", [0, -1, 1.5, "75", True])
def test_invalid_duration_tokens_rejected_before_loading(tokens):
    engine = MossTTSEngine(device="cpu")
    engine.load_model = Mock()
    with pytest.raises(ValueError, match="tokens must be a positive integer"):
        engine.synthesize("Hello", tokens=tokens)
    engine.load_model.assert_not_called()


@pytest.mark.parametrize("messages", [[], [None], [SimpleNamespace(audio_codes_list=[])]])
def test_missing_decoded_audio(messages):
    engine = _engine()
    engine.processor.decode.return_value = messages
    with pytest.raises(RuntimeError, match="returned no audio"):
        engine.synthesize("Hello")


@pytest.mark.parametrize("shape", [(0,), (1, 0), (1, 2, 3)])
def test_empty_or_invalid_waveform(shape):
    engine = _engine(torch.empty(shape))
    with pytest.raises(RuntimeError, match="empty or invalid audio waveform"):
        engine.synthesize("Hello")


def test_unload_releases_processor_codec_and_model(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    engine = _engine()
    processor_ref = weakref.ref(engine.processor)
    codec_ref = weakref.ref(engine.processor.audio_tokenizer)
    model_ref = weakref.ref(engine.model)

    engine.unload_model()
    gc.collect()

    assert engine.processor is None
    assert engine.model is None
    assert not engine._loaded
    assert processor_ref() is None
    assert codec_ref() is None
    assert model_ref() is None


def test_cuda_timing_synchronizes_and_scopes_attention(monkeypatch):
    engine = _engine(device="cuda:1")
    events = []
    timer_values = iter([10.0, 10.5])

    def timer():
        events.append("timer")
        return next(timer_values)

    def synchronize(device):
        assert device == "cuda:1"
        events.append("sync")

    def generate(**kwargs):
        assert torch.is_inference_mode_enabled()
        events.append("generate")
        return "generated"

    monkeypatch.setattr(moss.time, "perf_counter", timer)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    attention = Mock(return_value=nullcontext())
    monkeypatch.setattr(torch.nn.attention, "sdpa_kernel", attention)
    engine.model.generate.side_effect = generate

    result = engine.synthesize("Hello")

    assert events == ["sync", "timer", "generate", "sync", "timer"]
    assert result.inference_time_seconds == 0.5
    assert result.real_time_factor == 0.5
    backends = attention.call_args.args[0]
    assert torch.nn.attention.SDPBackend.CUDNN_ATTENTION not in backends
    engine.processor.decode.assert_called_once_with("generated")
