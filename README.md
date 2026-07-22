<div align="center">
<h2>UNITTS: One Python interface for open-source TTS models</h2>
<div>
    <a href="https://github.com/aynursusuz/unitts/actions/workflows/ci.yml" target="_blank">
        <img src="https://img.shields.io/github/actions/workflow/status/aynursusuz/unitts/ci.yml?branch=main&style=for-the-badge&labelColor=2D3748" alt="CI">
    </a>
    <a href="https://github.com/aynursusuz/unitts/blob/main/LICENSE" target="_blank">
        <img src="https://img.shields.io/badge/License-Apache_2.0-4ECDC4?style=for-the-badge&labelColor=2D3748" alt="License">
    </a>
    <a href="https://www.python.org/downloads/" target="_blank">
        <img src="https://img.shields.io/badge/Python-3.10+-45B7D1?style=for-the-badge&logo=python&logoColor=white&labelColor=2D3748" alt="Python 3.10+">
    </a>
</div>
</div>

## Overview

unitts gathers open-source text-to-speech models behind one Python interface. Install once and call any supported engine the same way; new engines are added one at a time.

## Installation

```bash
git clone https://github.com/aynursusuz/unitts.git
cd unitts
uv venv --python 3.12 && source .venv/bin/activate

# Base install (no engines)
uv pip install -e .

# Install only the engines you need
uv pip install -e ".[chatterbox]"
uv pip install -e ".[fish-audio]"
uv pip install -e ".[qwen3-tts]"
uv pip install -e ".[echo-tts]"
uv pip install -e ".[kokoro]"
uv pip install -e ".[f5-tts]"
uv pip install -e ".[voxcpm]"
```

> Chatterbox depends on `perth`, which still imports `pkg_resources`. On setuptools 80 or newer, also run `uv pip install "setuptools<80"`.
>
> Fish Audio pulls `fish-speech` from its GitHub repo (not on PyPI). `fish-speech` and `chatterbox-tts` currently pin different `torch` versions, so install them in separate environments. `fish-speech` also depends on `pyaudio`; on Debian/Ubuntu install its system headers first with `sudo apt-get install portaudio19-dev`.
>
> Qwen3-TTS pins `transformers==4.57.3` / `accelerate==1.12.0`, so install it in its own environment too. On CUDA you can optionally add FlashAttention 2 (`uv pip install flash-attn --no-build-isolation`) and pass `attn_implementation="flash_attention_2"` to `get_engine`.
>
> Echo-TTS needs a CUDA GPU (~8 GB VRAM). It depends on `torchcodec`, which loads the system FFmpeg libraries at runtime — install FFmpeg if it is missing. Its weights are non-commercial (CC-BY-NC-SA-4.0).

## Inference

```python
from unitts.engines import get_engine

engine = get_engine("chatterbox")
engine.synthesize_to_file("Hello world!", "out.wav")
```

Every engine exposes the same interface. Swap by changing the name:

```python
engine = get_engine("chatterbox")     # local, MIT
engine = get_engine("fish-audio")     # local, s2-pro weights, non-commercial
engine = get_engine("qwen3-tts")      # local, Apache-2.0, 10 languages
engine = get_engine("echo-tts")       # local, diffusion, MIT code / non-commercial weights
engine = get_engine("kokoro")         # local, Apache-2.0, 82M params, runs on CPU
engine = get_engine("f5-tts")         # local, voice cloning, MIT code / non-commercial weights
engine = get_engine("voxcpm")         # local, Apache-2.0, 30 languages, 48 kHz
```

First call to `fish-audio` downloads the 11 GB s2-pro checkpoint from HuggingFace into the default HF cache. Set `FISH_S2_PRO_DIR` to point at an existing local copy.

`qwen3-tts` defaults to the `Qwen3-TTS-12Hz-1.7B-CustomVoice` checkpoint, which picks a built-in speaker and accepts an optional natural-language `instruct` for emotion/style:

```python
engine = get_engine("qwen3-tts")
engine.synthesize_to_file(
    "Hello world!", "out.wav", speaker="Ryan", instruct="Cheerful and upbeat."
)

# Voice cloning uses the Base checkpoint (3-second clone from a reference clip):
clone = get_engine("qwen3-tts", model_path="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
clone.synthesize_to_file(
    "Hello world!", "clone.wav", ref_audio="ref.wav", ref_text="reference transcript"
)
```

Point at any released checkpoint with `model_path=` or the `QWEN3_TTS_MODEL` env var; `synthesize` routes to custom-voice, voice-design, or voice-clone generation based on which one you load. Weights download from HuggingFace on first use.

`echo-tts` is a diffusion model; raise `num_steps` for quality or lower it for speed, and pass `speaker_audio` to clone a voice:

```python
engine = get_engine("echo-tts", num_steps=40)
engine.synthesize_to_file("Hello world!", "out.wav")                             # built-in voice
engine.synthesize_to_file("Hello world!", "clone.wav", speaker_audio="ref.wav")  # cloned voice
```

### CLI

```bash
unitts list-engines
unitts synthesize "Hello world!" --engine chatterbox --output out.wav
unitts benchmark --engine chatterbox
```

`benchmark` writes one JSON to `benchmarks/results/<engine>.json` and one WAV to `benchmarks/audio_samples/<engine>.wav`.

## Engines

| Engine | Type | Voice cloning | License | Status |
|--------|------|:-------------:|---------|--------|
| [Chatterbox](https://github.com/resemble-ai/chatterbox) | local | yes | MIT | integrated |
| [Fish Audio s2-pro](https://huggingface.co/fishaudio/s2-pro) | local | yes | Fish Audio Research License | integrated |
| [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) | local | yes | Apache-2.0 | integrated |
| [Echo-TTS](https://github.com/FoxEngine-ai/echo-tts) | local | yes | CC-BY-NC-SA-4.0 (weights) | integrated |
| [Kokoro](https://github.com/hexgrad/kokoro) | local | no | Apache-2.0 | integrated |
| [F5-TTS](https://github.com/SWivid/F5-TTS) | local | yes | CC-BY-NC-4.0 (weights) | integrated |
| [VoxCPM2](https://github.com/OpenBMB/VoxCPM) | local | yes | Apache-2.0 | integrated |

## Benchmark

| Engine | RTF | Inference (s) | Audio (s) | VRAM (MB) | Sample rate |
|--------|-----|---------------|-----------|-----------|-------------|
| Chatterbox | 0.44 | 5.90 | 13.52 | 3,107 | 24,000 |
| Fish Audio s2-pro | 4.00 | 38.29 | 9.57 | 19,105 | 44,100 |
| Qwen3-TTS | 1.07 | 22.61 | 21.12 | 4,014 | 24,000 |
| Echo-TTS | **0.14** | 4.05 | 28.42 | 6,486 | 44,100 |

*RTF (real-time factor) = inference time / audio duration. Lower is faster.* Fish Audio measurements are without `--compile`; upstream documents ~5x speedup after kernel fusion. Full results: [`benchmarks/results/`](benchmarks/results/). Audio samples: [`benchmarks/audio_samples/`](benchmarks/audio_samples/). Chatterbox and Fish Audio were measured on an A100; Qwen3-TTS and Echo-TTS on an H100. Echo-TTS reaches the GPU only with a recent CUDA `torch` build; on older drivers it falls back to CPU.

## Adding an engine

1. Add `src/unitts/engines/<name>_engine.py`
2. Subclass `TTSEngine`, implement `load_model()` and `synthesize()`, decorate with `@register_engine`
3. Add the package to `pyproject.toml` extras
4. Add a unit test under `tests/unit/`

## License

unitts itself is Apache 2.0 (see [LICENSE](LICENSE)). Each integrated model keeps its own upstream license; by invoking an engine you agree to the terms of its model. Third-party model notices are listed in [NOTICE](NOTICE).

**Built with Fish Audio.** The `fish-audio` engine uses Fish Audio s2-pro weights under the Fish Audio Research License (non-commercial). Commercial use of that engine requires a separate license from Fish Audio.

The `qwen3-tts` engine uses Qwen3-TTS weights from the Qwen team at Alibaba Cloud, released under Apache 2.0.

The `echo-tts` engine uses Echo-TTS weights (`jordand/echo-tts-base`) under CC-BY-NC-SA-4.0 (non-commercial research); the `echo-tts` code is MIT. Commercial use of the weights is not permitted.

The `kokoro` engine uses Kokoro-82M weights (`hexgrad/Kokoro-82M`), released under Apache 2.0.

The `f5-tts` engine uses F5-TTS weights (`SWivid/F5-TTS`) under CC-BY-NC-4.0 (non-commercial); the `f5-tts` code is MIT. Commercial use of the weights is not permitted.

The `voxcpm` engine uses VoxCPM2 weights (`openbmb/VoxCPM2`) from OpenBMB, released under Apache 2.0.
