"""VoxCPM adapter (the open-source ``voxcpm`` package)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from unitts.engines.base import TTSEngine, TTSResult
from unitts.engines.registry import register_engine

_DEFAULT_MODEL = "openbmb/VoxCPM2"


@register_engine
class VoxCPMEngine(TTSEngine):
    """VoxCPM2: tokenizer-free TTS with voice cloning and free-form voice design.

    VoxCPM2 (OpenBMB, 2B parameters) covers 30 languages, outputs 48 kHz audio
    directly, and is Apache-2.0 licensed including the weights. Voices come
    three ways: the default voice, a voice designed from a natural-language
    description (prefix the text with ``"(description)"``), or a voice cloned
    from a reference clip.
    """

    name = "voxcpm"
    description = "VoxCPM2 (tokenizer-free, 30 languages, voice cloning + voice design)"
    url = "https://github.com/OpenBMB/VoxCPM"
    license = "Apache-2.0"
    languages = [
        "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi",
        "id", "it", "ja", "km", "ko", "lo", "ms", "my", "nl", "no",
        "pl", "pt", "ru", "sv", "sw", "th", "tl", "tr", "vi", "zh",
    ]  # fmt: skip
    supports_voice_cloning = True
    supports_streaming = True
    supports_emotion_control = True
    default_sample_rate = 48000

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL,
        load_denoiser: bool = False,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        **kwargs: Any,
    ) -> None:
        """Configure the engine.

        Args:
            model_path: HuggingFace id or local path of the checkpoint to load.
            load_denoiser: Whether to also load the audio denoiser used to clean
                noisy reference clips before cloning.
            cfg_value: Default classifier-free guidance scale.
            inference_timesteps: Default number of diffusion steps; higher is
                slower but higher quality.
            **kwargs: Forwarded to :class:`TTSEngine` (e.g. ``device``).
        """
        super().__init__(**kwargs)
        self.model_path = model_path
        self.load_denoiser = load_denoiser
        self.cfg_value = cfg_value
        self.inference_timesteps = inference_timesteps

    def load_model(self) -> None:
        """Load the VoxCPM model; weights download from HuggingFace on first use."""
        from voxcpm import VoxCPM

        self.model = VoxCPM.from_pretrained(self.model_path, load_denoiser=self.load_denoiser)

    def synthesize(
        self,
        text: str,
        ref_audio: str | None = None,
        prompt_audio: str | None = None,
        prompt_text: str | None = None,
        cfg_value: float | None = None,
        inference_timesteps: int | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> TTSResult:
        """Synthesize ``text`` with VoxCPM.

        Args:
            text: The text to read aloud. Prefix with a parenthesized description
                (e.g. ``"(a calm, deep male narrator) Hello"``) for voice design.
            ref_audio: Optional reference clip for controllable voice cloning
                (timbre is kept, style can still be guided).
            prompt_audio: Optional reference clip for exact voice cloning in
                audio-continuation mode; pair with ``prompt_text``.
            prompt_text: Transcript of ``prompt_audio``.
            cfg_value: Guidance scale; defaults to the engine's ``cfg_value``.
            inference_timesteps: Diffusion steps; defaults to the engine's
                ``inference_timesteps``.
            seed: Random seed for reproducible output.
            **kwargs: Extra options forwarded to ``VoxCPM.generate``.

        Returns:
            The synthesized audio and timing metadata.
        """
        self.ensure_loaded()

        generate_kwargs: dict[str, Any] = {
            "cfg_value": self.cfg_value if cfg_value is None else cfg_value,
            "inference_timesteps": (
                self.inference_timesteps if inference_timesteps is None else inference_timesteps
            ),
            **kwargs,
        }
        if ref_audio is not None:
            generate_kwargs["reference_wav_path"] = ref_audio
        if prompt_audio is not None:
            generate_kwargs["prompt_wav_path"] = prompt_audio
        if prompt_text is not None:
            generate_kwargs["prompt_text"] = prompt_text
        if seed is not None:
            generate_kwargs["seed"] = seed

        start = time.perf_counter()
        wav = self.model.generate(text=text, **generate_kwargs)
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
                "model_path": self.model_path,
                "inference_timesteps": generate_kwargs["inference_timesteps"],
                "voice_cloned": ref_audio is not None or prompt_audio is not None,
                "seed": seed,
            },
        )
