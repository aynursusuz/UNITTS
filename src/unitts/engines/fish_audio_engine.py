"""Fish Audio s2-pro adapter (HuggingFace weights)."""

from __future__ import annotations

import os
import site
import time
from pathlib import Path
from typing import Any

import numpy as np

from unitts.engines.base import TTSEngine, TTSResult
from unitts.engines.registry import register_engine

_HF_REPO = "fishaudio/s2-pro"
_PRECISION_MAP = {"bfloat16": "bfloat16", "float16": "float16", "half": "float16"}


def _ensure_project_root_marker() -> None:
    """fish_speech imports call pyrootutils.setup_root with a `.project-root`
    indicator. When installed from PyPI or a git URL there is no such file in
    site-packages, so we touch one the first time the adapter loads.
    """
    for path in site.getsitepackages():
        marker = Path(path) / ".project-root"
        try:
            marker.touch(exist_ok=True)
            return
        except OSError:
            continue


@register_engine
class FishAudioEngine(TTSEngine):
    """Fish Audio s2-pro: multilingual, 44.1 kHz, voice cloning from a reference id.

    The weights are released under a non-commercial research license.
    """

    name = "fish-audio"
    description = "Fish Audio s2-pro (HF weights, non-commercial)"
    url = "https://huggingface.co/fishaudio/s2-pro"
    license = "Fish Audio Research License"
    languages = [
        "en",
        "zh",
        "ja",
        "ko",
        "es",
        "pt",
        "ar",
        "ru",
        "fr",
        "de",
        "it",
        "tr",
        "nl",
        "pl",
    ]
    supports_voice_cloning = True
    supports_streaming = True
    default_sample_rate = 44100

    def __init__(
        self,
        checkpoint_dir: str | Path | None = None,
        precision: str = "bfloat16",
        use_compile: bool = False,
        **kwargs: Any,
    ) -> None:
        """Configure the engine.

        Args:
            checkpoint_dir: Local s2-pro checkpoint directory. Falls back to the
                ``FISH_S2_PRO_DIR`` environment variable, then to a HuggingFace
                download on first load.
            precision: Inference precision: ``"bfloat16"``, ``"float16"`` or ``"half"``.
            use_compile: Compile the model for faster inference at the cost of a
                slower first call.
            **kwargs: Forwarded to :class:`TTSEngine` (e.g. ``device``).
        """
        super().__init__(**kwargs)
        env_dir = os.environ.get("FISH_S2_PRO_DIR")
        self.checkpoint_dir = (
            Path(checkpoint_dir) if checkpoint_dir else (Path(env_dir) if env_dir else None)
        )
        if precision not in _PRECISION_MAP:
            raise ValueError(f"precision must be one of {list(_PRECISION_MAP)}, got {precision!r}")
        self.precision_name = _PRECISION_MAP[precision]
        self.use_compile = use_compile

    def load_model(self) -> None:
        """Download the s2-pro checkpoint if needed and build the inference engine."""
        import torch
        from huggingface_hub import snapshot_download

        _ensure_project_root_marker()

        if self.checkpoint_dir is None or not self.checkpoint_dir.exists():
            self.checkpoint_dir = Path(snapshot_download(repo_id=_HF_REPO))

        from fish_speech.inference_engine import TTSInferenceEngine
        from fish_speech.models.dac.inference import load_model as load_decoder
        from fish_speech.models.text2semantic.inference import launch_thread_safe_queue

        precision_dtype = torch.bfloat16 if self.precision_name == "bfloat16" else torch.half

        llama_queue = launch_thread_safe_queue(
            checkpoint_path=self.checkpoint_dir,
            device=self.device,
            precision=precision_dtype,
            compile=self.use_compile,
        )
        decoder = load_decoder(
            config_name="modded_dac_vq",
            checkpoint_path=self.checkpoint_dir / "codec.pth",
            device=self.device,
        )
        self.model = TTSInferenceEngine(
            llama_queue=llama_queue,
            decoder_model=decoder,
            compile=self.use_compile,
            precision=precision_dtype,
        )

    def synthesize(
        self,
        text: str,
        reference_id: str | None = None,
        max_new_tokens: int = 1024,
        chunk_length: int = 200,
        top_p: float = 0.7,
        repetition_penalty: float = 1.5,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> TTSResult:
        """Synthesize ``text`` with Fish Audio s2-pro.

        Args:
            text: The text to read aloud.
            reference_id: Optional id of a stored reference voice to clone.
            max_new_tokens: Maximum number of audio tokens to generate.
            chunk_length: Number of characters per synthesis chunk.
            top_p: Nucleus sampling probability.
            repetition_penalty: Penalty applied to repeated tokens.
            temperature: Sampling temperature; higher is more varied.
            **kwargs: Accepted for interface compatibility; currently unused.

        Returns:
            The synthesized audio and timing metadata.

        Raises:
            RuntimeError: If the model produces no audio.
        """
        from fish_speech.utils.schema import ServeTTSRequest

        self.ensure_loaded()

        req = ServeTTSRequest(
            text=text,
            references=[],
            reference_id=reference_id,
            max_new_tokens=max_new_tokens,
            chunk_length=chunk_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            format="wav",
        )

        start = time.perf_counter()
        results = list(self.model.inference(req))
        elapsed = time.perf_counter() - start

        final = next((r for r in results if r.code == "final"), None)
        if final is None:
            err = next((r for r in results if r.code == "error"), None)
            raise RuntimeError(f"fish-audio inference failed: {err.error if err else 'no output'}")

        sample_rate, audio = final.audio
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        duration = len(audio) / sample_rate

        return TTSResult(
            audio=audio,
            sample_rate=sample_rate,
            duration_seconds=duration,
            inference_time_seconds=elapsed,
            real_time_factor=elapsed / duration if duration else 0.0,
            engine_name=self.name,
            text=text,
            metadata={"reference_id": reference_id, "precision": self.precision_name},
        )
