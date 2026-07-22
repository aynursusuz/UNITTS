"""NeuTTS adapter (the open-source ``neutts`` package)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from unitts.engines.base import TTSEngine, TTSResult
from unitts.engines.registry import register_engine

_DEFAULT_BACKBONE = "neuphonic/neutts-air"
_DEFAULT_CODEC = "neuphonic/neucodec"


@register_engine
class NeuTTSEngine(TTSEngine):
    """NeuTTS: Neuphonic's on-device TTS with instant voice cloning on CPU.

    NeuTTS always speaks in a cloned voice: every call needs a short reference
    clip (``ref_audio``) and its transcript (``ref_text``). The default
    ``neutts-air`` backbone is Apache-2.0; pass ``backbone_repo`` to use the
    newer multilingual ``neutts-nano`` checkpoints (NeuTTS Open License).
    """

    name = "neutts"
    description = "NeuTTS (on-device, CPU voice cloning)"
    url = "https://github.com/neuphonic/neutts"
    license = "Apache-2.0 (neutts-air weights)"
    languages = ["en"]
    supports_voice_cloning = True
    default_sample_rate = 24000

    def __init__(
        self,
        backbone_repo: str = _DEFAULT_BACKBONE,
        codec_repo: str = _DEFAULT_CODEC,
        **kwargs: Any,
    ) -> None:
        """Configure the engine.

        Args:
            backbone_repo: HuggingFace repo of the language-model backbone.
            codec_repo: HuggingFace repo of the audio codec.
            **kwargs: Forwarded to :class:`TTSEngine` (e.g. ``device``).
        """
        super().__init__(**kwargs)
        self.backbone_repo = backbone_repo
        self.codec_repo = codec_repo

    def load_model(self) -> None:
        """Load the NeuTTS backbone and codec; weights download on first use."""
        from neutts import NeuTTS

        self.model = NeuTTS(
            backbone_repo=self.backbone_repo,
            backbone_device=self.device,
            codec_repo=self.codec_repo,
            codec_device=self.device,
        )

    def synthesize(
        self,
        text: str,
        ref_audio: str | None = None,
        ref_text: str | None = None,
        **kwargs: Any,
    ) -> TTSResult:
        """Synthesize ``text`` in the voice of ``ref_audio``.

        Args:
            text: The text to read aloud.
            ref_audio: Path to the reference clip to clone the voice from
                (required; a few seconds of clean speech works best).
            ref_text: Transcript of ``ref_audio`` (required).
            **kwargs: Extra options forwarded to ``NeuTTS.infer``.

        Returns:
            The synthesized audio and timing metadata.

        Raises:
            ValueError: If ``ref_audio`` or ``ref_text`` is missing.
        """
        self.ensure_loaded()
        if ref_audio is None or ref_text is None:
            raise ValueError(
                "neutts is a voice-cloning model: pass ref_audio (a short reference "
                "clip) and ref_text (its transcript)."
            )

        start = time.perf_counter()
        ref_codes = self.model.encode_reference(ref_audio)
        wav = self.model.infer(text, ref_codes, ref_text, **kwargs)
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
                "backbone_repo": self.backbone_repo,
                "ref_audio": ref_audio,
            },
        )
