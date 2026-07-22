"""F5-TTS adapter (the open-source ``f5-tts`` package)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from unitts.engines.base import TTSEngine, TTSResult
from unitts.engines.registry import register_engine

_DEFAULT_MODEL = "F5TTS_v1_Base"


@register_engine
class F5TTSEngine(TTSEngine):
    """F5-TTS: zero-shot voice cloning with flow matching.

    F5-TTS always speaks in a cloned voice: every call needs a short reference
    clip (``ref_audio``). Pass its transcript as ``ref_text``, or leave it out
    to have the reference transcribed automatically with Whisper (slower on
    the first call, which downloads the ASR model).

    The F5-TTS code is MIT licensed; the default ``F5TTS_v1_Base`` weights are
    CC-BY-NC-4.0 (non-commercial, due to the Emilia training data).
    """

    name = "f5-tts"
    description = "F5-TTS (flow matching, zero-shot voice cloning)"
    url = "https://github.com/SWivid/F5-TTS"
    license = "CC-BY-NC-4.0 (weights), MIT (code)"
    languages = ["en", "zh"]
    supports_voice_cloning = True
    default_sample_rate = 24000

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        ckpt_file: str = "",
        vocab_file: str = "",
        nfe_step: int = 32,
        speed: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Configure the engine.

        Args:
            model: Model config name, e.g. ``"F5TTS_v1_Base"`` or ``"E2TTS_Base"``.
            ckpt_file: Optional local checkpoint path; defaults to downloading
                the named model from HuggingFace.
            vocab_file: Optional local vocab path.
            nfe_step: Default number of flow-matching denoising steps; higher is
                slower but higher quality.
            speed: Default speaking speed multiplier.
            **kwargs: Forwarded to :class:`TTSEngine` (e.g. ``device``).
        """
        super().__init__(**kwargs)
        self.model_name = model
        self.ckpt_file = ckpt_file
        self.vocab_file = vocab_file
        self.nfe_step = nfe_step
        self.speed = speed

    def load_model(self) -> None:
        """Load the F5-TTS model and vocoder; weights download on first use."""
        from f5_tts.api import F5TTS

        self.model = F5TTS(
            model=self.model_name,
            ckpt_file=self.ckpt_file,
            vocab_file=self.vocab_file,
            device=self.device,
        )

    def synthesize(
        self,
        text: str,
        ref_audio: str | None = None,
        ref_text: str | None = None,
        nfe_step: int | None = None,
        speed: float | None = None,
        seed: int | None = None,
        remove_silence: bool = False,
        **kwargs: Any,
    ) -> TTSResult:
        """Synthesize ``text`` in the voice of ``ref_audio``.

        Args:
            text: The text to read aloud.
            ref_audio: Path to the reference clip to clone the voice from
                (required; a few seconds of clean speech works best).
            ref_text: Transcript of ``ref_audio``. Omit to auto-transcribe the
                reference with Whisper.
            nfe_step: Denoising steps; defaults to the engine's ``nfe_step``.
            speed: Speaking speed multiplier; defaults to the engine's ``speed``.
            seed: Random seed for reproducible output.
            remove_silence: Whether to trim long silences from the output.
            **kwargs: Extra options forwarded to ``F5TTS.infer`` (e.g.
                ``cfg_strength``, ``cross_fade_duration``, ``target_rms``).

        Returns:
            The synthesized audio and timing metadata.

        Raises:
            ValueError: If ``ref_audio`` is missing.
        """
        self.ensure_loaded()
        if ref_audio is None:
            raise ValueError(
                "f5-tts is a voice-cloning model: pass ref_audio (a short reference "
                "clip), and optionally ref_text with its transcript."
            )

        start = time.perf_counter()
        wav, sample_rate, _spec = self.model.infer(
            ref_file=ref_audio,
            ref_text=ref_text or "",
            gen_text=text,
            nfe_step=self.nfe_step if nfe_step is None else nfe_step,
            speed=self.speed if speed is None else speed,
            seed=seed,
            remove_silence=remove_silence,
            show_info=lambda *args, **kw: None,
            **kwargs,
        )
        elapsed = time.perf_counter() - start

        audio = np.asarray(wav, dtype=np.float32).squeeze()
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
                "model": self.model_name,
                "nfe_step": self.nfe_step if nfe_step is None else nfe_step,
                "seed": getattr(self.model, "seed", seed),
                "ref_audio": ref_audio,
            },
        )
