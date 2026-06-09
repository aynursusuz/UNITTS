"""Chatterbox TTS adapter."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from unitts.engines.base import TTSEngine, TTSResult
from unitts.engines.registry import register_engine


@register_engine
class ChatterboxEngine(TTSEngine):
    """Resemble AI's Chatterbox: a fast English model with prompt-based voice cloning."""

    name = "chatterbox"
    description = "Chatterbox TTS"
    url = "https://github.com/resemble-ai/chatterbox"
    license = "MIT"
    languages = ["en"]
    supports_voice_cloning = True
    default_sample_rate = 24000

    def load_model(self) -> None:
        """Download the pretrained Chatterbox weights and read its sample rate."""
        from chatterbox.tts import ChatterboxTTS

        self.model = ChatterboxTTS.from_pretrained(device=self.device)
        self.default_sample_rate = getattr(self.model, "sr", self.default_sample_rate)

    def synthesize(
        self,
        text: str,
        audio_prompt_path: str | None = None,
        **kwargs: Any,
    ) -> TTSResult:
        """Synthesize ``text`` with Chatterbox.

        Args:
            text: The text to read aloud.
            audio_prompt_path: Optional reference clip to clone the voice from.
            **kwargs: Extra options forwarded to ``ChatterboxTTS.generate``.

        Returns:
            The synthesized audio and timing metadata.
        """
        self.ensure_loaded()

        start = time.perf_counter()
        wav = self.model.generate(text, audio_prompt_path=audio_prompt_path, **kwargs)
        elapsed = time.perf_counter() - start

        audio = wav.detach().cpu().numpy().astype(np.float32).squeeze()
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
        )
