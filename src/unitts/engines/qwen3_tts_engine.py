"""Qwen3-TTS adapter (the open-source ``qwen-tts`` package)."""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np

from unitts.engines.base import TTSEngine, TTSResult
from unitts.engines.registry import register_engine

_DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
_DTYPE_MAP = {
    "bfloat16": "bfloat16",
    "float16": "float16",
    "half": "float16",
    "float32": "float32",
}


@register_engine
class Qwen3TTSEngine(TTSEngine):
    """Qwen3-TTS 12Hz, wrapping all three checkpoint families behind one engine.

    Each released checkpoint exposes a different generation entry point, and
    :meth:`synthesize` dispatches on the loaded checkpoint's ``tts_model_type``:

    * ``*-CustomVoice`` -> a built-in ``speaker`` plus an optional ``instruct``
      for emotion or style. This is the default checkpoint, so ``synthesize(text)``
      works with no extra arguments.
    * ``*-VoiceDesign`` -> a voice invented from an ``instruct`` description.
    * ``*-Base``        -> a 3-second voice clone from ``ref_audio`` (and
      ``ref_text`` for in-context cloning).

    Select a checkpoint with ``model_path`` or the ``QWEN3_TTS_MODEL`` environment
    variable.
    """

    name = "qwen3-tts"
    description = "Qwen3-TTS 12Hz (custom voice / voice design / voice clone)"
    url = "https://github.com/QwenLM/Qwen3-TTS"
    license = "Apache-2.0"
    languages = ["zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"]
    supports_voice_cloning = True
    supports_streaming = True
    supports_emotion_control = True
    default_sample_rate = 24000

    def __init__(
        self,
        model_path: str | None = None,
        speaker: str = "Ryan",
        language: str = "Auto",
        dtype: str = "bfloat16",
        attn_implementation: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Configure the engine.

        Args:
            model_path: HuggingFace id or local path of the checkpoint to load.
                Falls back to the ``QWEN3_TTS_MODEL`` environment variable, then to
                the 1.7B CustomVoice checkpoint.
            speaker: Default built-in speaker for CustomVoice checkpoints.
            language: Default language, or ``"Auto"`` to detect it from the text.
            dtype: Compute dtype: ``"bfloat16"``, ``"float16"``, ``"half"`` or
                ``"float32"``.
            attn_implementation: Optional attention backend, e.g.
                ``"flash_attention_2"``.
            **kwargs: Forwarded to :class:`TTSEngine` (e.g. ``device``).
        """
        super().__init__(**kwargs)
        self.model_path = model_path or os.environ.get("QWEN3_TTS_MODEL") or _DEFAULT_MODEL
        self.speaker = speaker
        self.language = language
        if dtype not in _DTYPE_MAP:
            raise ValueError(f"dtype must be one of {list(_DTYPE_MAP)}, got {dtype!r}")
        self.dtype_name = _DTYPE_MAP[dtype]
        self.attn_implementation = attn_implementation
        self.model_type: str | None = None

    def load_model(self) -> None:
        """Load the checkpoint and record which generation mode it supports."""
        import torch
        from qwen_tts import Qwen3TTSModel

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.dtype_name]

        load_kwargs: dict[str, Any] = {"device_map": self.device, "dtype": torch_dtype}
        if self.attn_implementation:
            load_kwargs["attn_implementation"] = self.attn_implementation

        self.model = Qwen3TTSModel.from_pretrained(self.model_path, **load_kwargs)
        # Each checkpoint is one of: "custom_voice", "voice_design", "base".
        self.model_type = getattr(self.model.model, "tts_model_type", "custom_voice")

    def synthesize(
        self,
        text: str,
        speaker: str | None = None,
        instruct: str | None = None,
        language: str | None = None,
        ref_audio: Any = None,
        ref_text: str | None = None,
        x_vector_only_mode: bool = False,
        **kwargs: Any,
    ) -> TTSResult:
        """Synthesize ``text`` with the loaded Qwen3-TTS checkpoint.

        Which arguments apply depends on the checkpoint (see the class docstring):
        ``speaker``/``instruct`` for CustomVoice, ``instruct`` for VoiceDesign, and
        ``ref_audio``/``ref_text`` for Base voice cloning.

        Args:
            text: The text to read aloud.
            speaker: Built-in speaker name (CustomVoice only); defaults to the
                engine's ``speaker``.
            instruct: Natural-language style or voice description (CustomVoice and
                VoiceDesign).
            language: Language name, or ``"Auto"`` to detect it; defaults to the
                engine's ``language``.
            ref_audio: Reference clip for voice cloning (Base only): a path, URL,
                ``(waveform, sample_rate)`` tuple, or list thereof.
            ref_text: Transcript of ``ref_audio`` for in-context cloning (Base only).
            x_vector_only_mode: Clone from the speaker embedding only, ignoring
                ``ref_text`` (Base only).
            **kwargs: Generation options forwarded to the model (e.g.
                ``max_new_tokens``, ``top_p``, ``temperature``).

        Returns:
            The synthesized audio and timing metadata.

        Raises:
            ValueError: If required inputs for the loaded checkpoint are missing.
        """
        self.ensure_loaded()
        lang = language or self.language

        start = time.perf_counter()
        if self.model_type == "base":
            if ref_audio is None:
                raise ValueError(
                    "qwen3-tts Base checkpoint does voice cloning: pass ref_audio "
                    "(and ref_text for in-context cloning)."
                )
            wavs, sample_rate = self.model.generate_voice_clone(
                text=text,
                language=lang,
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=x_vector_only_mode,
                **kwargs,
            )
        elif self.model_type == "voice_design":
            if not instruct:
                raise ValueError(
                    "qwen3-tts VoiceDesign checkpoint needs `instruct` describing the voice."
                )
            wavs, sample_rate = self.model.generate_voice_design(
                text=text, instruct=instruct, language=lang, **kwargs
            )
        else:  # custom_voice
            wavs, sample_rate = self.model.generate_custom_voice(
                text=text,
                speaker=speaker or self.speaker,
                language=lang,
                instruct=instruct,
                **kwargs,
            )
        elapsed = time.perf_counter() - start

        audio = np.asarray(wavs[0], dtype=np.float32).squeeze()
        duration = len(audio) / sample_rate if sample_rate else 0.0

        return TTSResult(
            audio=audio,
            sample_rate=sample_rate,
            duration_seconds=duration,
            inference_time_seconds=elapsed,
            real_time_factor=elapsed / duration if duration else 0.0,
            engine_name=self.name,
            text=text,
            metadata={
                "model_path": self.model_path,
                "mode": self.model_type,
                "speaker": (speaker or self.speaker) if self.model_type == "custom_voice" else None,
                "language": lang,
            },
        )
