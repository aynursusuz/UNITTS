"""Base class for all TTS engine adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class TTSResult:
    """Result from a TTS inference call."""

    audio: np.ndarray
    sample_rate: int
    duration_seconds: float
    inference_time_seconds: float
    real_time_factor: float
    engine_name: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_faster_than_realtime(self) -> bool:
        return self.real_time_factor < 1.0


class TTSEngine(ABC):
    """Abstract base class for TTS engines."""

    name: str = "base"
    description: str = ""
    url: str = ""
    license: str = ""
    languages: list[str] = []
    supports_voice_cloning: bool = False
    supports_streaming: bool = False
    supports_emotion_control: bool = False
    default_sample_rate: int = 24000

    def __init__(self, device: str = "auto", **kwargs: Any) -> None:
        if device == "auto":
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model = None
        self._loaded = False

    @abstractmethod
    def load_model(self) -> None:
        """Load the model into memory. Called once before inference."""
        ...

    @abstractmethod
    def synthesize(self, text: str, **kwargs: Any) -> TTSResult:
        """Synthesize speech from text.

        Args:
            text: The text to synthesize.
            **kwargs: Engine-specific parameters (voice, speaker, emotion, etc.)

        Returns:
            TTSResult with audio data and metadata.
        """
        ...

    def synthesize_to_file(self, text: str, output_path: str | Path, **kwargs: Any) -> TTSResult:
        """Synthesize speech and save to a WAV file."""
        import soundfile as sf

        result = self.synthesize(text, **kwargs)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), result.audio, result.sample_rate)
        return result

    def ensure_loaded(self) -> None:
        """Ensure the model is loaded, loading it if necessary."""
        if not self._loaded:
            self.load_model()
            self._loaded = True

    def unload_model(self) -> None:
        """Unload the model from memory."""
        self.model = None
        self._loaded = False
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_vram_usage_mb(self) -> float | None:
        """Get current VRAM usage in MB, or None if not on GPU."""
        import torch

        if self.device == "cuda" and torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return None

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} loaded={self._loaded}>"
