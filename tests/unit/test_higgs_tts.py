"""Higgs HTTP adapter contracts; no external server, GPU or weights are needed."""

import base64
import io
import json
from urllib.error import HTTPError, URLError

import numpy as np
import pytest
import soundfile as sf

from unitts.engines import get_engine
from unitts.engines import higgs_tts_engine as higgs
from unitts.engines.higgs_tts_engine import HiggsTTSEngine

MODEL_ID = "bosonai/higgs-tts-3-4b"


def _wav(samples=24000, channels=1, sample_rate=24000):
    buffer = io.BytesIO()
    shape = (samples,) if channels == 1 else (samples, channels)
    sf.write(buffer, np.full(shape, 0.25, dtype=np.float32), sample_rate, format="WAV")
    return buffer.getvalue()


class _Response(io.BytesIO):
    pass


def _server(monkeypatch, *, models=None, speech=None):
    requests = []
    model_ids = [MODEL_ID] if models is None else models

    def urlopen(request, timeout):
        requests.append((request, timeout))
        if request.full_url.endswith("/health"):
            return _Response(b'{"status":"healthy"}')
        if request.full_url.endswith("/v1/models"):
            return _Response(json.dumps({"data": [{"id": m} for m in model_ids]}).encode())
        assert request.full_url.endswith("/v1/audio/speech")
        return _Response(_wav() if speech is None else speech)

    monkeypatch.setattr(higgs, "urlopen", urlopen)
    return requests


def test_metadata_and_constructor_are_lazy(monkeypatch):
    def forbid_network(*args, **kwargs):
        raise AssertionError("Construction must not contact a server")

    monkeypatch.setattr(higgs, "urlopen", forbid_network)
    engine = get_engine("higgs-tts", device="cuda")
    assert isinstance(engine, HiggsTTSEngine)
    assert engine.model_path == MODEL_ID
    assert engine.device == "server"
    assert engine.get_vram_usage_mb() is None
    assert "tr" in engine.languages
    assert "ja" in engine.languages
    assert engine.supports_voice_cloning
    assert engine.supports_emotion_control
    assert not engine.supports_streaming
    assert "Non-Commercial" in engine.license
    assert not engine._loaded


def test_environment_defaults_and_explicit_overrides(monkeypatch):
    monkeypatch.setenv("HIGGS_TTS_BASE_URL", "https://higgs.example/v1/")
    monkeypatch.setenv("HIGGS_TTS_MODEL", "served-alias")
    monkeypatch.setenv("HIGGS_TTS_API_KEY", "environment-token")
    engine = HiggsTTSEngine()
    assert engine.base_url == "https://higgs.example"
    assert engine.model_path == "served-alias"
    assert engine.api_key == "environment-token"
    engine = HiggsTTSEngine(base_url="http://localhost:9000", model_path=MODEL_ID, api_key="")
    assert engine.base_url == "http://localhost:9000"
    assert engine.model_path == MODEL_ID
    assert engine.api_key == ""


@pytest.mark.parametrize(
    "base_url", ["file:///tmp/server", "ftp://server", "http://", "http://a:b@x"]
)
def test_invalid_server_url(base_url):
    with pytest.raises(ValueError, match="HTTP"):
        HiggsTTSEngine(base_url=base_url)


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True])
def test_invalid_timeout(timeout):
    with pytest.raises(ValueError, match="timeout"):
        HiggsTTSEngine(timeout=timeout)


def test_load_checks_health_and_exact_model_once(monkeypatch):
    requests = _server(monkeypatch)
    engine = HiggsTTSEngine(api_key="test-token", timeout=25)
    engine.ensure_loaded()
    engine.ensure_loaded()
    assert [request.full_url for request, _ in requests] == [
        "http://127.0.0.1:8000/health",
        "http://127.0.0.1:8000/v1/models",
    ]
    assert all(
        request.get_header("Authorization") == "Bearer test-token" for request, _ in requests
    )
    assert all(timeout == 25 for _, timeout in requests)
    assert engine.model == MODEL_ID


def test_wrong_model_is_not_treated_as_higgs(monkeypatch):
    requests = _server(monkeypatch, models=["OpenMOSS-Team/MOSS-TTS-v1.5"])
    engine = HiggsTTSEngine()
    with pytest.raises(RuntimeError, match="does not serve"):
        engine.synthesize("Merhaba.")
    assert len(requests) == 2
    assert not engine._loaded
    assert engine.model is None


@pytest.mark.parametrize("listing", [b"not-json", b"{}", b'{"data":null}'])
def test_invalid_model_listing(monkeypatch, listing):
    engine = HiggsTTSEngine()
    monkeypatch.setattr(engine, "_request", lambda path: b"ok" if path == "/health" else listing)
    with pytest.raises(RuntimeError, match="invalid /v1/models"):
        engine.ensure_loaded()
    assert not engine._loaded


def test_turkish_request_preserves_text_and_inline_tags(monkeypatch):
    requests = _server(monkeypatch)
    ticks = iter([10.0, 10.5])
    monkeypatch.setattr(higgs.time, "perf_counter", lambda: next(ticks))
    text = "<|emotion:contentment|>Merhaba dünya. <|prosody:pause|>Bugün nasılsın?"
    result = HiggsTTSEngine().synthesize(text, temperature=0.7, top_p=0.9, seed=42)
    request, _ = requests[-1]
    payload = json.loads(request.data)
    assert request.get_method() == "POST"
    assert request.get_header("Content-type") == "application/json"
    assert payload == {
        "model": MODEL_ID,
        "input": text,
        "voice": "default",
        "response_format": "wav",
        "stream": False,
        "temperature": 0.7,
        "top_k": 50,
        "max_new_tokens": 2048,
        "top_p": 0.9,
        "seed": 42,
    }
    assert result.audio.dtype == np.float32
    assert result.audio.shape == (24000,)
    assert result.duration_seconds == 1.0
    assert result.inference_time_seconds == 0.5
    assert result.real_time_factor == 0.5
    assert result.engine_name == "higgs-tts"
    assert result.metadata["timing_scope"] == "http_round_trip_including_audio_decode"


def test_reference_uploaded_as_data_uri_not_server_path(monkeypatch, tmp_path):
    requests = _server(monkeypatch)
    reference = tmp_path / "speaker.wav"
    data = _wav(samples=12000)
    reference.write_bytes(data)
    result = HiggsTTSEngine().synthesize(
        "Merhaba.", ref_audio=reference, ref_text="Referans cümlesi."
    )
    payload = json.loads(requests[-1][0].data)
    upload = payload["references"][0]
    assert upload["audio_path"].startswith("data:audio/")
    assert base64.b64decode(upload["audio_path"].split(",", 1)[1]) == data
    assert str(reference) not in upload["audio_path"]
    assert upload["text"] == "Referans cümlesi."
    assert result.metadata["voice_cloning"]


@pytest.mark.parametrize("samples,channels,sample_rate", [(12000, 1, 12000), (48000, 2, 48000)])
def test_audio_shape_and_rate_come_from_response(monkeypatch, samples, channels, sample_rate):
    _server(monkeypatch, speech=_wav(samples, channels, sample_rate))
    result = HiggsTTSEngine().synthesize("Hello.")
    assert result.audio.shape == ((samples,) if channels == 1 else (samples, channels))
    assert result.sample_rate == sample_rate
    assert result.duration_seconds == 1.0


@pytest.mark.parametrize(
    "text,kwargs",
    [
        (" ", {}),
        ("Hello", {"language": "tr"}),
        ("Hello", {"stream": True}),
        ("Hello", {"ref_text": "Without reference audio"}),
    ],
)
def test_invalid_inputs_rejected_before_contacting_server(monkeypatch, text, kwargs):
    requests = _server(monkeypatch)
    with pytest.raises(ValueError):
        HiggsTTSEngine().synthesize(text, **kwargs)
    assert requests == []


def test_http_error_has_actionable_message(monkeypatch):
    def fail(request, timeout):
        raise HTTPError(request.full_url, 503, "Unavailable", {}, io.BytesIO(b"model loading"))

    monkeypatch.setattr(higgs, "urlopen", fail)
    with pytest.raises(RuntimeError, match="HTTP 503: model loading"):
        HiggsTTSEngine().ensure_loaded()


def test_connection_failure_mentions_server_setup(monkeypatch):
    def fail(request, timeout):
        raise URLError("connection refused")

    monkeypatch.setattr(higgs, "urlopen", fail)
    with pytest.raises(RuntimeError, match="unitts serve --engine higgs-tts"):
        HiggsTTSEngine().ensure_loaded()


@pytest.mark.parametrize("speech,match", [(b"{}", "decodable WAV"), (_wav(samples=0), "empty")])
def test_invalid_audio_response(monkeypatch, speech, match):
    _server(monkeypatch, speech=speech)
    with pytest.raises(RuntimeError, match=match):
        HiggsTTSEngine().synthesize("Hello.")


def test_unload_does_not_send_shutdown_or_touch_local_gpu(monkeypatch):
    requests = _server(monkeypatch)
    engine = HiggsTTSEngine(device="cuda")
    engine.ensure_loaded()
    engine.unload_model()
    assert len(requests) == 2
    assert engine.model is None
    assert not engine._loaded
    assert engine.get_vram_usage_mb() is None
    engine.ensure_loaded()
    assert len(requests) == 4
