"""MOSS-TTS v1.5 adapter using the official Hugging Face remote-code API."""

from __future__ import annotations

import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np

from unitts.engines.base import TTSEngine, TTSResult
from unitts.engines.registry import register_engine

_DEFAULT_MODEL = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
_LANGUAGES = {
    "zh": "Chinese",
    "yue": "Cantonese",
    "en": "English",
    "ar": "Arabic",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "fi": "Finnish",
    "fr": "French",
    "de": "German",
    "el": "Greek",
    "he": "Hebrew",
    "hi": "Hindi",
    "hu": "Hungarian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "mk": "Macedonian",
    "ms": "Malay",
    "fa": "Persian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "es": "Spanish",
    "sw": "Swahili",
    "sv": "Swedish",
    "tl": "Tagalog",
    "th": "Thai",
    "tr": "Turkish",
    "vi": "Vietnamese",
}


def _language_name(language: str | None) -> str | None:
    if language is None or language.lower() == "auto":
        return None
    for code, name in _LANGUAGES.items():
        if language.lower() in (code, name.lower()):
            return name
    raise ValueError(f"Unsupported MOSS-TTS language {language!r}; use {list(_LANGUAGES)}")


@register_engine
class MossTTSEngine(TTSEngine):
    """MOSS-TTS v1.5: 4B Local Transformer (default) or 8B Delay.

    Both support 31 languages and optional reference-audio voice cloning.
    The upstream processor owns a separate audio-tokenizer model. Its native
    output is 48 kHz stereo for Local v1.5 and 24 kHz mono for Delay v1.5.
    This adapter returns complete audio, not streaming chunks.
    """

    name = "moss-tts"
    description = "MOSS-TTS v1.5 (4B local / 8B delay, 31 languages)"
    url = "https://github.com/OpenMOSS/MOSS-TTS"
    license = "Apache-2.0"
    languages = list(_LANGUAGES)
    supports_voice_cloning = True
    default_sample_rate = 48000

    def __init__(
        self,
        model_path: str | None = None,
        language: str | None = None,
        dtype: str = "auto",
        attn_implementation: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model_path = model_path or os.environ.get("MOSS_TTS_MODEL") or _DEFAULT_MODEL
        self.language = _language_name(language)
        if dtype not in ("auto", "bfloat16", "float16", "float32"):
            raise ValueError("dtype must be auto, bfloat16, float16 or float32")
        self.dtype_name = dtype
        self.attn_implementation = attn_implementation
        self.processor = None

    def load_model(self) -> None:
        """Load upstream model code, weights and the checkpoint's audio codec."""
        import torch
        from transformers import AutoModel, AutoProcessor

        is_cuda = self.device.startswith("cuda")
        dtype_name = self.dtype_name
        if dtype_name == "auto":
            dtype_name = "bfloat16" if is_cuda else "float32"
        attention = self.attn_implementation or ("sdpa" if is_cuda else "eager")

        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.processor.audio_tokenizer = self.processor.audio_tokenizer.to(self.device)
        self.processor.audio_tokenizer.eval()
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            dtype=getattr(torch, dtype_name),
            attn_implementation=attention,
        ).to(self.device)
        self.model.eval()
        self.default_sample_rate = int(self.processor.model_config.sampling_rate)

    def synthesize(
        self,
        text: str,
        language: str | None = None,
        ref_audio: str | Path | None = None,
        tokens: int | None = None,
        **kwargs: Any,
    ) -> TTSResult:
        """Generate audio, optionally cloning a reference clip.

        ``language`` accepts a code (e.g. ``tr``) or English name (``Turkish``).
        Set it for non-English/Chinese text. ``tokens`` controls target duration
        in upstream audio frames; other kwargs go to ``model.generate`` (e.g.
        ``max_new_tokens`` and ``audio_temperature``).
        """
        import torch

        if not text.strip():
            raise ValueError("text must not be empty")
        lang = self.language if language is None else _language_name(language)
        if tokens is not None and (
            isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0
        ):
            raise ValueError("tokens must be a positive integer")
        self.ensure_loaded()
        is_cuda = self.device.startswith("cuda")
        if is_cuda:
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()

        # Upstream excludes cuDNN SDPA for this model. Scope that workaround to
        # this call so it does not change attention settings for other engines.
        attention_context = nullcontext()
        if is_cuda and self.attn_implementation in (None, "sdpa"):
            from torch.nn.attention import SDPBackend, sdpa_kernel

            attention_context = sdpa_kernel(
                [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
            )

        with torch.inference_mode(), attention_context:
            message = self.processor.build_user_message(
                text=text,
                language=lang,
                reference=[str(ref_audio)] if ref_audio is not None else None,
                tokens=tokens,
            )
            batch = self.processor([[message]], mode="generation")
            generation_kwargs = {
                "max_new_tokens": 4096,
                "do_sample": True,
                "audio_temperature": 1.7,
                "audio_top_p": 0.8,
                "audio_top_k": 25,
                "audio_repetition_penalty": 1.0,
                **kwargs,
            }
            outputs = self.model.generate(
                input_ids=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
                **generation_kwargs,
            )
            messages = self.processor.decode(outputs)
            if not messages or messages[0] is None or not messages[0].audio_codes_list:
                raise RuntimeError("MOSS-TTS returned no audio")
            waveform = messages[0].audio_codes_list[0]
            audio = waveform.detach().to(device="cpu", dtype=torch.float32).numpy()

        # soundfile expects [samples, channels], upstream returns [channels, samples].
        if audio.ndim == 2:
            audio = audio[0] if audio.shape[0] == 1 else audio.T
        if audio.ndim not in (1, 2) or audio.size == 0:
            raise RuntimeError("MOSS-TTS returned an empty or invalid audio waveform")
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        if is_cuda:
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - start
        sample_rate = self.default_sample_rate
        duration = len(audio) / sample_rate
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
                "language": lang,
                "voice_cloning": ref_audio is not None,
                "channels": audio.shape[1] if audio.ndim == 2 else 1,
            },
        )

    def unload_model(self) -> None:
        """Release both the language model and processor-owned audio codec."""
        self.processor = None
        super().unload_model()
