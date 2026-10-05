"""Chatterbox's multilingual adapter, separate from the English engine."""

from __future__ import annotations

import inspect
import time
from typing import Any

import numpy as np

from unitts.engines.base import TTSEngine, TTSResult
from unitts.engines.registry import register_engine


@register_engine
class ChatterboxMultilingualEngine(TTSEngine):
    """23-language Chatterbox with optional reference-audio voice cloning.

    The default uses the checkpoint selected by the installed package (V2 in
    ``chatterbox-tts==0.1.7``). Set ``t3_model="v3"`` only with an upstream
    Chatterbox installation that supports explicit checkpoint selection.
    """

    name = "chatterbox-multilingual"
    description = "Chatterbox Multilingual (23 languages, voice cloning)"
    url = "https://github.com/resemble-ai/chatterbox"
    license = "MIT"
    languages = [
        "ar",
        "da",
        "de",
        "el",
        "en",
        "es",
        "fi",
        "fr",
        "he",
        "hi",
        "it",
        "ja",
        "ko",
        "ms",
        "nl",
        "no",
        "pl",
        "pt",
        "ru",
        "sv",
        "sw",
        "tr",
        "zh",
    ]
    supports_voice_cloning = True
    supports_emotion_control = True
    default_sample_rate = 24000

    def __init__(
        self,
        language: str = "en",
        t3_model: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.language = self._validate_language(language)
        self.t3_model = t3_model

    def _validate_language(self, language: str) -> str:
        language = language.strip().lower()
        if language not in self.languages:
            raise ValueError(
                f"Unsupported Chatterbox language {language!r}. "
                f"Use one of: {', '.join(self.languages)}"
            )
        return language

    def load_model(self) -> None:
        """Load lazily; fail clearly if checkpoint selection is unavailable."""
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        load_kwargs: dict[str, Any] = {"device": self.device}
        if self.t3_model is not None:
            parameters = inspect.signature(ChatterboxMultilingualTTS.from_pretrained).parameters
            if "t3_model" not in parameters:
                raise ValueError(
                    "This Chatterbox installation does not support t3_model selection. "
                    "Omit t3_model to use its default multilingual checkpoint, or install "
                    "an upstream Chatterbox version with t3_model support to select V3."
                )
            load_kwargs["t3_model"] = self.t3_model

        self.model = ChatterboxMultilingualTTS.from_pretrained(**load_kwargs)
        self.default_sample_rate = self.model.sr

    def _synchronize_cuda(self) -> None:
        if str(self.device).split(":", 1)[0] == "cuda":
            import torch

            torch.cuda.synchronize(self.device)

    def synthesize(
        self,
        text: str,
        audio_prompt_path: str | None = None,
        language: str | None = None,
        language_id: str | None = None,
        **kwargs: Any,
    ) -> TTSResult:
        """Generate audio using a language code such as ``tr``.

        ``language_id`` is an alias for ``language``, matching the upstream API.
        Other options, including ``exaggeration``, ``cfg_weight`` and
        ``temperature``, are forwarded to Chatterbox. Its PerTh watermark is
        preserved. Reference clips should match the target language to reduce
        accent transfer.
        """
        if not text.strip():
            raise ValueError("text must not be empty")
        lang = self._validate_language(language) if language is not None else self.language
        if language_id is not None:
            alias = self._validate_language(language_id)
            if language is not None and alias != lang:
                raise ValueError("language and language_id must match when both are supplied")
            lang = alias
        self.ensure_loaded()

        self._synchronize_cuda()
        start = time.perf_counter()
        wav = self.model.generate(
            text,
            language_id=lang,
            audio_prompt_path=audio_prompt_path,
            **kwargs,
        )
        self._synchronize_cuda()
        elapsed = time.perf_counter() - start

        audio = wav.detach().cpu().numpy().astype(np.float32).reshape(-1)
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
                "model_path": "ResembleAI/chatterbox",
                "t3_model": self.t3_model,
                "language": lang,
                "voice_cloned": audio_prompt_path is not None,
            },
        )
