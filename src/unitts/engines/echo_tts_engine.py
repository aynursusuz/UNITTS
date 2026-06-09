"""Echo-TTS adapter (the open-source ``echo-tts`` package)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from unitts.engines.base import TTSEngine, TTSResult
from unitts.engines.registry import register_engine

_DTYPE_MAP = {
    "bfloat16": "bfloat16",
    "float16": "float16",
    "half": "float16",
    "float32": "float32",
}


@register_engine
class EchoTTSEngine(TTSEngine):
    """Echo-TTS: a diffusion text-to-speech model with speaker-reference voice cloning.

    The ``echo-tts`` code is MIT licensed; the ``jordand/echo-tts-base`` weights are
    released for non-commercial research use only (CC-BY-NC-SA-4.0). A CUDA GPU with
    at least 8 GB of VRAM is recommended.
    """

    name = "echo-tts"
    description = "Echo-TTS (diffusion, voice cloning)"
    url = "https://github.com/FoxEngine-ai/echo-tts"
    license = "CC-BY-NC-SA-4.0 (weights), MIT (code)"
    languages = ["en"]
    supports_voice_cloning = True
    default_sample_rate = 44100

    def __init__(
        self,
        dtype: str = "bfloat16",
        num_steps: int = 40,
        sequence_length: int = 640,
        use_compile: bool = False,
        **kwargs: Any,
    ) -> None:
        """Configure the engine.

        Args:
            dtype: Compute dtype: ``"bfloat16"``, ``"float16"``, ``"half"`` or ``"float32"``.
            num_steps: Number of diffusion sampling steps; higher is slower but
                higher quality.
            sequence_length: Maximum output length in latent frames (~30s at 640).
            use_compile: Whether to ``torch.compile`` the model (slower first call,
                faster afterwards).
            **kwargs: Forwarded to :class:`TTSEngine` (e.g. ``device``).
        """
        super().__init__(**kwargs)
        if dtype not in _DTYPE_MAP:
            raise ValueError(f"dtype must be one of {list(_DTYPE_MAP)}, got {dtype!r}")
        self.dtype_name = _DTYPE_MAP[dtype]
        self.num_steps = num_steps
        self.sequence_length = sequence_length
        self.use_compile = use_compile

    def load_model(self) -> None:
        """Download the Echo-TTS weights and build the model, autoencoder and PCA state."""
        import torch
        from echo_tts import EchoTTS

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.dtype_name]
        self.model = EchoTTS(device=self.device, dtype=torch_dtype, compile=self.use_compile)

    def synthesize(
        self,
        text: str,
        speaker_audio: str | None = None,
        seed: int = 0,
        num_steps: int | None = None,
        sequence_length: int | None = None,
        **kwargs: Any,
    ) -> TTSResult:
        """Synthesize ``text`` with Echo-TTS.

        Args:
            text: The text to read aloud. Turns may be prefixed with speaker tags
                such as ``"[S1] ..."`` as documented upstream.
            speaker_audio: Optional path to a reference clip to clone the voice from.
            seed: Random seed for reproducible output.
            num_steps: Diffusion steps; defaults to the engine's ``num_steps``.
            sequence_length: Maximum output length; defaults to the engine's
                ``sequence_length``.
            **kwargs: Extra options forwarded to ``EchoTTS.synthesize`` (e.g.
                ``cfg_scale_text``, ``cfg_scale_speaker``, ``truncation_factor``).

        Returns:
            The synthesized audio and timing metadata.
        """
        self.ensure_loaded()

        start = time.perf_counter()
        audio_tensor, sample_rate = self.model.synthesize(
            text=text,
            speaker_audio=speaker_audio,
            seed=seed,
            num_steps=self.num_steps if num_steps is None else num_steps,
            sequence_length=self.sequence_length if sequence_length is None else sequence_length,
            **kwargs,
        )
        elapsed = time.perf_counter() - start

        audio = audio_tensor.detach().cpu().numpy().astype(np.float32).squeeze()
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
                "seed": seed,
                "num_steps": self.num_steps if num_steps is None else num_steps,
                "voice_cloned": speaker_audio is not None,
            },
        )
