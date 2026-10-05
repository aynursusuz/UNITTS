"""Higgs TTS 3 adapter for an independently managed SGLang-Omni server."""

from __future__ import annotations

import base64
import io
import json
import math
import mimetypes
import os
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import numpy as np

from unitts.engines.base import TTSEngine, TTSResult
from unitts.engines.registry import register_engine

_DEFAULT_MODEL = "bosonai/higgs-tts-3-4b"
_GENERATION_OPTIONS = {"temperature", "top_p", "top_k", "max_new_tokens", "seed"}


@register_engine
class HiggsTTSEngine(TTSEngine):
    """Use the official Higgs TTS 3 HTTP API and return complete WAV audio.

    Start SGLang-Omni separately with ``bosonai/higgs-tts-3-4b``. ``load_model``
    verifies readiness and the served model; ``unload_model`` only disconnects
    this adapter. Server GPU allocation and process lifetime stay with its owner.
    Languages are inferred from the input text; no language field is sent.
    """

    name = "higgs-tts"
    description = "Higgs TTS 3 4B (100+ languages, SGLang-Omni server)"
    url = "https://huggingface.co/bosonai/higgs-tts-3-4b"
    license = "Boson Higgs TTS 3 Research and Non-Commercial License (Creator Use Grant)"
    # Model-card metadata, plus Japanese and Thai in its supported-language table.
    languages = [
        "af",
        "ar",
        "as",
        "ast",
        "az",
        "ba",
        "be",
        "bg",
        "bn",
        "bs",
        "ca",
        "ceb",
        "ckb",
        "cs",
        "cy",
        "da",
        "de",
        "el",
        "en",
        "eo",
        "es",
        "et",
        "eu",
        "fa",
        "fi",
        "fr",
        "ga",
        "gl",
        "gu",
        "ha",
        "he",
        "hi",
        "hr",
        "ht",
        "hu",
        "hy",
        "id",
        "is",
        "it",
        "ja",
        "jv",
        "ka",
        "kab",
        "kam",
        "kea",
        "kk",
        "kn",
        "ko",
        "ky",
        "la",
        "lb",
        "lg",
        "ln",
        "lt",
        "luo",
        "lv",
        "mhr",
        "mi",
        "mk",
        "ml",
        "mn",
        "mr",
        "ms",
        "mt",
        "ne",
        "nl",
        "no",
        "nso",
        "ny",
        "oc",
        "om",
        "pa",
        "pl",
        "ps",
        "pt",
        "ro",
        "ru",
        "rw",
        "sd",
        "sk",
        "sl",
        "sn",
        "so",
        "sq",
        "sr",
        "sv",
        "sw",
        "ta",
        "te",
        "tg",
        "th",
        "tl",
        "tr",
        "ug",
        "uk",
        "umb",
        "ur",
        "uz",
        "vi",
        "xh",
        "zh",
        "zu",
    ]
    supports_voice_cloning = True
    supports_emotion_control = True
    default_sample_rate = 24000

    def __init__(
        self,
        base_url: str | None = None,
        model_path: str | None = None,
        api_key: str | None = None,
        timeout: float = 180.0,
        device: str = "auto",
        **kwargs: Any,
    ) -> None:
        """Configure the client without importing or allocating a GPU runtime.

        ``base_url`` defaults to ``HIGGS_TTS_BASE_URL`` or http://127.0.0.1:8000.
        ``model_path`` must match an ID advertised by ``/v1/models``; it defaults
        to ``HIGGS_TTS_MODEL`` or the official 4B checkpoint. ``api_key`` defaults
        to ``HIGGS_TTS_API_KEY``. ``device`` is accepted for registry compatibility;
        the running server, rather than this client, controls the actual device.
        """
        del device
        super().__init__(device="server", **kwargs)
        self.base_url = (
            base_url or os.environ.get("HIGGS_TTS_BASE_URL") or "http://127.0.0.1:8000"
        ).rstrip("/")
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]
        parsed = urlsplit(self.base_url)
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("base_url must be an HTTP(S) server URL without credentials or query")
        if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be a positive finite number of seconds")
        self.timeout = float(timeout)
        self.model_path = model_path or os.environ.get("HIGGS_TTS_MODEL") or _DEFAULT_MODEL
        self.api_key = api_key if api_key is not None else os.environ.get("HIGGS_TTS_API_KEY")

    def _request(self, path: str, payload: dict[str, Any] | None = None) -> bytes:
        headers = {"Accept": "application/json" if payload is None else "audio/wav"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = None
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(self.base_url + path, data=data, headers=headers)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return response.read()
        except HTTPError as exc:
            detail = exc.read(1000).decode("utf-8", errors="replace")
            raise RuntimeError(f"Higgs TTS server returned HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(
                f"Cannot reach Higgs TTS server at {self.base_url}: {exc}. "
                "Start SGLang-Omni with the configured Higgs model first."
            ) from exc

    def load_model(self) -> None:
        """Verify the external server is ready and serves the requested model."""
        self._request("/health")
        try:
            listing = json.loads(self._request("/v1/models"))
            entries = listing["data"]
            available = [entry["id"] for entry in entries]
        except (ValueError, KeyError, TypeError) as exc:
            raise RuntimeError("Higgs TTS server returned an invalid /v1/models response") from exc
        if self.model_path not in available:
            raise RuntimeError(
                f"Higgs TTS server does not serve {self.model_path!r}; "
                f"advertised models: {available}. Set model_path to the served Higgs model ID."
            )
        self.model = self.model_path

    def synthesize(
        self,
        text: str,
        ref_audio: str | Path | None = None,
        ref_text: str | None = None,
        **kwargs: Any,
    ) -> TTSResult:
        """Synthesize text with optional local reference audio and transcript.

        Reference bytes travel as a data URI, so the server need not share the
        client's filesystem. Inline emotion/prosody tokens pass through unchanged.
        Supported generation options are ``temperature``, ``top_p``, ``top_k``,
        ``max_new_tokens`` and ``seed``. Timing includes the HTTP round trip and
        audio decoding, but excludes server readiness checks and reference loading.
        """
        import soundfile as sf

        if not text.strip():
            raise ValueError("text must not be empty")
        unsupported = kwargs.keys() - _GENERATION_OPTIONS
        if unsupported:
            raise ValueError(
                f"Unsupported Higgs TTS options: {sorted(unsupported)}. "
                "Language is inferred from text; use inline tags for style and emotion."
            )
        if ref_text is not None and ref_audio is None:
            raise ValueError("ref_text requires ref_audio")

        payload: dict[str, Any] = {
            "model": self.model_path,
            "input": text,
            "voice": "default",
            "response_format": "wav",
            "stream": False,
            "temperature": 0.8,
            "top_k": 50,
            "max_new_tokens": 2048,
            **kwargs,
        }
        if ref_audio is not None:
            reference_path = Path(ref_audio).expanduser()
            reference_bytes = reference_path.read_bytes()
            if not reference_bytes:
                raise ValueError("ref_audio must not be empty")
            media_type = mimetypes.guess_type(str(reference_path))[0] or "application/octet-stream"
            encoded = base64.b64encode(reference_bytes).decode("ascii")
            data_uri = f"data:{media_type};base64,{encoded}"
            reference = {"audio_path": data_uri}
            if ref_text is not None:
                reference["text"] = ref_text
            payload["references"] = [reference]

        self.ensure_loaded()
        start = time.perf_counter()
        response = self._request("/v1/audio/speech", payload)
        try:
            audio, sample_rate = sf.read(io.BytesIO(response), dtype="float32")
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError("Higgs TTS server did not return decodable WAV audio") from exc
        elapsed = time.perf_counter() - start
        if audio.ndim not in (1, 2) or not audio.size or sample_rate <= 0:
            raise RuntimeError("Higgs TTS server returned empty or invalid audio")
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        duration = len(audio) / sample_rate
        self.default_sample_rate = sample_rate
        return TTSResult(
            audio=audio,
            sample_rate=sample_rate,
            duration_seconds=duration,
            inference_time_seconds=elapsed,
            real_time_factor=elapsed / duration,
            engine_name=self.name,
            text=text,
            metadata={
                "model_path": self.model_path,
                "backend": "sglang-omni",
                "device": "server",
                "timing_scope": "http_round_trip_including_audio_decode",
                "voice_cloning": ref_audio is not None,
            },
        )

    def get_vram_usage_mb(self) -> None:
        """The HTTP API does not expose this model's VRAM allocation."""
        return None

    def unload_model(self) -> None:
        """Reset client state without stopping or unloading the external server."""
        self.model = None
        self._loaded = False
