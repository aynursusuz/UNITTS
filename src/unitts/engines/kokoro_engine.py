"""Kokoro adapter (the open-source ``kokoro`` package)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from unitts.engines.base import TTSEngine, TTSResult
from unitts.engines.registry import register_engine

_DEFAULT_REPO = "hexgrad/Kokoro-82M"

# KPipeline language codes -> ISO codes, for the engine metadata.
_LANG_CODES = {
    "a": "en",  # American English
    "b": "en",  # British English
    "e": "es",
    "f": "fr",
    "h": "hi",
    "i": "it",
    "j": "ja",
    "p": "pt",
    "z": "zh",
}


@register_engine
class KokoroEngine(TTSEngine):
    """Kokoro: an 82M-parameter TTS model that runs faster than real time on CPU.

    Kokoro is the lightweight engine in unitts: Apache-2.0 weights, ~330 MB
    checkpoint, dozens of built-in voices across 9 languages, no GPU required.
    It has no voice cloning; pick from the built-in voices instead.

    English text needs ``espeak-ng`` installed on the system as a
    grapheme-to-phoneme fallback; Japanese and Chinese need the
    ``misaki[ja]`` / ``misaki[zh]`` extras.
    """

    name = "kokoro"
    description = "Kokoro-82M (lightweight, CPU real-time, 9 languages)"
    url = "https://github.com/hexgrad/kokoro"
    license = "Apache-2.0"
    languages = ["en", "es", "fr", "hi", "it", "ja", "pt", "zh"]
    supports_voice_cloning = False
    default_sample_rate = 24000

    def __init__(
        self,
        lang_code: str = "a",
        voice: str = "af_heart",
        speed: float = 1.0,
        repo_id: str = _DEFAULT_REPO,
        **kwargs: Any,
    ) -> None:
        """Configure the engine.

        Args:
            lang_code: KPipeline language code: ``"a"`` (American English),
                ``"b"`` (British English), ``"e"`` (es), ``"f"`` (fr), ``"h"`` (hi),
                ``"i"`` (it), ``"j"`` (ja), ``"p"`` (pt-br) or ``"z"`` (zh).
            voice: Default built-in voice, e.g. ``"af_heart"`` or ``"am_adam"``.
                The first letter matches the language code.
            speed: Default speaking speed multiplier.
            repo_id: HuggingFace repo to load the weights from.
            **kwargs: Forwarded to :class:`TTSEngine` (e.g. ``device``).
        """
        super().__init__(**kwargs)
        if lang_code not in _LANG_CODES:
            raise ValueError(f"lang_code must be one of {list(_LANG_CODES)}, got {lang_code!r}")
        self.lang_code = lang_code
        self.voice = voice
        self.speed = speed
        self.repo_id = repo_id

    def load_model(self) -> None:
        """Build the Kokoro pipeline; weights download from HuggingFace on first use."""
        from kokoro import KPipeline

        self.model = KPipeline(lang_code=self.lang_code, repo_id=self.repo_id, device=self.device)

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float | None = None,
        split_pattern: str = r"\n+",
        **kwargs: Any,
    ) -> TTSResult:
        """Synthesize ``text`` with Kokoro.

        Args:
            text: The text to read aloud. Long text is split on ``split_pattern``
                and synthesized chunk by chunk.
            voice: Built-in voice name; defaults to the engine's ``voice``.
            speed: Speaking speed multiplier; defaults to the engine's ``speed``.
            split_pattern: Regex used to split ``text`` into chunks.
            **kwargs: Extra options forwarded to the Kokoro pipeline.

        Returns:
            The synthesized audio (all chunks concatenated) and timing metadata.
        """
        self.ensure_loaded()
        chosen_voice = voice or self.voice

        start = time.perf_counter()
        chunks: list[np.ndarray] = []
        for _graphemes, _phonemes, chunk in self.model(
            text,
            voice=chosen_voice,
            speed=self.speed if speed is None else speed,
            split_pattern=split_pattern,
            **kwargs,
        ):
            if hasattr(chunk, "detach"):
                chunk = chunk.detach().cpu().numpy()
            chunks.append(np.asarray(chunk, dtype=np.float32).squeeze())
        elapsed = time.perf_counter() - start

        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
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
                "lang_code": self.lang_code,
                "chunks": len(chunks),
            },
        )
