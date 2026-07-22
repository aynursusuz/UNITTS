"""Supertonic adapter (the open-source ``supertonic`` package)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from unitts.engines.base import TTSEngine, TTSResult
from unitts.engines.registry import register_engine


@register_engine
class SupertonicEngine(TTSEngine):
    """Supertonic: an ONNX-based on-device TTS model from Supertone.

    Supertonic runs on CPU (or GPU) through ONNX Runtime and covers 31
    languages with ten built-in voices (``M1``-``M5``, ``F1``-``F5``).
    The sample code is MIT licensed; the weights are OpenRAIL-M.
    """

    name = "supertonic"
    description = "Supertonic (ONNX on-device, 31 languages)"
    url = "https://github.com/supertone-inc/supertonic"
    license = "OpenRAIL-M (weights), MIT (code)"
    languages = [
        "ar", "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr",
        "hi", "hr", "hu", "id", "it", "ja", "ko", "lt", "lv", "nl", "pl",
        "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "vi",
    ]  # fmt: skip
    supports_voice_cloning = False
    default_sample_rate = 44100

    def __init__(
        self,
        voice: str = "M1",
        lang: str = "en",
        total_steps: int = 8,
        speed: float = 1.05,
        **kwargs: Any,
    ) -> None:
        """Configure the engine.

        Args:
            voice: Default built-in voice: ``"M1"``-``"M5"`` or ``"F1"``-``"F5"``.
            lang: Default language code, or ``"na"`` for language-agnostic input.
            total_steps: Default denoising steps (5-12); higher is slower but
                higher quality.
            speed: Default speaking speed (0.7-2.0).
            **kwargs: Forwarded to :class:`TTSEngine` (e.g. ``device``).
        """
        super().__init__(**kwargs)
        self.voice = voice
        self.lang = lang
        self.total_steps = total_steps
        self.speed = speed

    def load_model(self) -> None:
        """Build the Supertonic pipeline; assets download from HuggingFace on first use."""
        from supertonic import TTS

        self.model = TTS(auto_download=True)

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        lang: str | None = None,
        total_steps: int | None = None,
        speed: float | None = None,
        **kwargs: Any,
    ) -> TTSResult:
        """Synthesize ``text`` with Supertonic.

        Args:
            text: The text to read aloud.
            voice: Built-in voice name; defaults to the engine's ``voice``.
            lang: Language code; defaults to the engine's ``lang``.
            total_steps: Denoising steps; defaults to the engine's ``total_steps``.
            speed: Speaking speed; defaults to the engine's ``speed``.
            **kwargs: Extra options forwarded to ``TTS.synthesize``.

        Returns:
            The synthesized audio and timing metadata.
        """
        self.ensure_loaded()
        chosen_voice = voice or self.voice

        start = time.perf_counter()
        style = self.model.get_voice_style(voice_name=chosen_voice)
        wav, _duration = self.model.synthesize(
            text=text,
            lang=lang or self.lang,
            voice_style=style,
            total_steps=self.total_steps if total_steps is None else total_steps,
            speed=self.speed if speed is None else speed,
            **kwargs,
        )
        elapsed = time.perf_counter() - start

        audio = np.asarray(wav, dtype=np.float32).squeeze()
        sample_rate = self.default_sample_rate
        duration = len(audio) / sample_rate

        return TTSResult(
            audio=audio,
            sample_rate=sample_rate,
            duration_seconds=duration,
            inference_time_seconds=elapsed,
            real_time_factor=elapsed / duration if duration else 0.0,
            engine_name=self.name,
            text=text,
            metadata={
                "voice": chosen_voice,
                "lang": lang or self.lang,
                "total_steps": self.total_steps if total_steps is None else total_steps,
            },
        )
