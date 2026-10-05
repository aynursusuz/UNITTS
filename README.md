<div align="center">
<h2>UNITTS: One Python interface for TTS models</h2>
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

unitts gathers local and self-hosted text-to-speech models behind one Python interface. Call supported engines the same way; new engines are added one at a time. Model licenses vary, including open-source and research-only weights.

## Installation

```bash
git clone https://github.com/aynursusuz/unitts.git
cd unitts
uv venv --python 3.12 && source .venv/bin/activate

# Base install (no model weights)
uv pip install -e .

# Install only the engines you need
uv pip install -e ".[chatterbox]"
uv pip install -e ".[fish-audio]"
uv pip install -e ".[qwen3-tts]"
uv pip install -e ".[echo-tts]"
uv pip install -e ".[kokoro]"
uv pip install -e ".[supertonic]"
uv pip install -e ".[neutts]"
uv pip install --torch-backend cu128 -e ".[moss-tts]"
# higgs-tts uses the base install and a separate server (see below).
```

> Chatterbox depends on `perth`, which still imports `pkg_resources`. On setuptools 80 or newer, also run `uv pip install "setuptools<80"`.
>
> Fish Audio pulls `fish-speech` from its GitHub repo (not on PyPI). `fish-speech` and `chatterbox-tts` currently pin different `torch` versions, so install them in separate environments. `fish-speech` also depends on `pyaudio`; on Debian/Ubuntu install its system headers first with `sudo apt-get install portaudio19-dev`.
>
> Qwen3-TTS pins `transformers==4.57.3` / `accelerate==1.12.0`, so install it in its own environment too. On CUDA you can optionally add FlashAttention 2 (`uv pip install flash-attn --no-build-isolation`) and pass `attn_implementation="flash_attention_2"` to `get_engine`.
>
> Echo-TTS needs a CUDA GPU (~8 GB VRAM). It depends on `torchcodec`, which loads the system FFmpeg libraries at runtime — install FFmpeg if it is missing. Its weights are non-commercial (CC-BY-NC-SA-4.0).

> Use a separate Python 3.12 environment and FFmpeg for MOSS-TTS. Higgs uses a separate SGLang-Omni server. Expand the setup below for the tested GPU configuration.

<details>
<summary>GPU environment setup</summary>

From the UNITTS checkout, use separate environments for the two runtime stacks. The driver must support CUDA 12.8 for MOSS or CUDA 13.0 for Higgs. Weights download on first use.

```bash
# MOSS-TTS (install FFmpeg with your OS package manager if missing)
uv venv --python 3.12 .venv-moss
source .venv-moss/bin/activate
uv pip install --torch-backend cu128 -e ".[moss-tts]"

# Higgs server: run in its own terminal
# Keep the server source outside the UNITTS checkout.
git clone https://github.com/sgl-project/sglang-omni.git ../sglang-omni-higgs
git -C ../sglang-omni-higgs checkout 3d4eb6e49096aa42e7f8f4e72f028823b2d85b73
uv venv --python 3.12 .venv-higgs-server
source .venv-higgs-server/bin/activate
uv pip install --prerelease=allow --torch-backend cu130 -e ../sglang-omni-higgs
sgl-omni serve --model-path bosonai/higgs-tts-3-4b \
    --model-name bosonai/higgs-tts-3-4b \
    --host 127.0.0.1 --port 8001 \
    --mem-fraction-static 0.55 --max-running-requests 8 --cuda-graph-max-bs 8
```

Run UNITTS from its own environment in another terminal, and use one engine at a time on a shared GPU. Stop the Higgs server with Ctrl+C when finished; `engine.unload_model()` only resets its client.

</details>

## Inference

```python
from unitts.engines import get_engine

engine = get_engine("chatterbox")
engine.synthesize_to_file("Hello world!", "out.wav")
```

Every engine exposes the same interface. Swap by changing the name:

```python
engine = get_engine("chatterbox")     # local, MIT
engine = get_engine("moss-tts")       # local, Apache-2.0, 31 languages including Turkish
engine = get_engine("higgs-tts", base_url="http://127.0.0.1:8001")  # self-hosted
engine = get_engine("fish-audio")     # local, s2-pro weights, non-commercial
engine = get_engine("qwen3-tts")      # local, Apache-2.0, 10 languages
engine = get_engine("echo-tts")       # local, diffusion, MIT code / non-commercial weights
engine = get_engine("kokoro")         # local, Apache-2.0, 82M params, runs on CPU
engine = get_engine("supertonic")     # local, ONNX on-device, 31 languages, runs on CPU
engine = get_engine("neutts")         # local, voice cloning on CPU, Apache-2.0
```

`moss-tts` defaults to MOSS-TTS v1.5 8B (31 languages); `higgs-tts` uses Higgs TTS 3 4B (100+ languages). Both support Turkish. Select the language for MOSS; Higgs infers it from the text and requires the server configured above.

```python
engine = get_engine("moss-tts", device="cuda", language="tr")
# Or, with the Higgs server running:
# engine = get_engine("higgs-tts", base_url="http://127.0.0.1:8001")
engine.synthesize_to_file("Merhaba dünya!", "out.wav")
engine.unload_model()
```

For voice cloning, pass `ref_audio="ref.wav"` to either engine's synthesis call. Higgs also accepts `ref_text` with the reference recording's transcript.

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

# Higgs: connect to the server configured above
HIGGS_TTS_BASE_URL=http://127.0.0.1:8001 unitts benchmark --engine higgs-tts
```

`benchmark` writes one JSON to `benchmarks/results/<engine>.json` and one WAV to `benchmarks/audio_samples/<engine>.wav`.

The CLI supports basic synthesis; use the Python API above for language selection, reference audio and model-specific generation options.

## Engines

| Engine | Type | Voice cloning | License | Status |
|--------|------|:-------------:|---------|--------|
| [Chatterbox](https://github.com/resemble-ai/chatterbox) | local | yes | MIT | integrated |
| [MOSS-TTS v1.5](https://github.com/OpenMOSS/MOSS-TTS) | local | yes | Apache-2.0 | integrated |
| [Higgs TTS 3 4B](https://huggingface.co/bosonai/higgs-tts-3-4b) | self-hosted HTTP | yes | Boson Research and Non-Commercial | integrated |
| [Fish Audio s2-pro](https://huggingface.co/fishaudio/s2-pro) | local | yes | Fish Audio Research License | integrated |
| [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) | local | yes | Apache-2.0 | integrated |
| [Echo-TTS](https://github.com/FoxEngine-ai/echo-tts) | local | yes | CC-BY-NC-SA-4.0 (weights) | integrated |
| [Kokoro](https://github.com/hexgrad/kokoro) | local | no | Apache-2.0 | integrated |
| [Supertonic](https://github.com/supertone-inc/supertonic) | local | no | OpenRAIL-M (weights) | integrated |
| [NeuTTS](https://github.com/neuphonic/neutts) | local | yes | Apache-2.0 | integrated |

## Benchmark

| Engine | GPU | RTF | Inference (s) | Audio (s) | VRAM (MiB) | Sample rate |
|--------|-----|-----|---------------|-----------|------------|-------------|
| [Chatterbox](benchmarks/results/chatterbox.json) | A100 | 0.44 | 5.90 | 13.52 | 3,107 | 24,000 |
| [Fish Audio s2-pro](benchmarks/results/fish-audio.json) | A100 | 4.00 | 38.29 | 9.57 | 19,105 | 44,100 |
| [Qwen3-TTS](benchmarks/results/qwen3-tts.json) | H100 | 1.07 | 22.61 | 21.12 | 4,014 | 24,000 |
| [Echo-TTS](benchmarks/results/echo-tts.json) | H100 | 0.14 | 4.05 | 28.42 | 6,486 | 44,100 |
| [MOSS-TTS v1.5](benchmarks/results/gpu-smoke/moss-v1.5.json) | RTX 6000 Ada | 0.487 | 2.57 | 5.28 | 23,168* | 24,000 |
| [Higgs TTS 3 4B](benchmarks/results/gpu-smoke/higgs-tts.json) | RTX 6000 Ada | 0.284 | 1.58 | 5.56 | — | 24,000 |

*RTF = inference time / audio duration; lower is faster.* GPUs and input texts differ. MOSS and Higgs are single warm Turkish samples, excluding model loading; Higgs timing includes HTTP and WAV decoding. These measurements are not a controlled speed or quality ranking. Fish Audio was measured without `--compile`.

VRAM values are allocated-memory snapshots; `*` marks peak allocation for the MOSS sample. `—` means server VRAM was not measured. Row links contain full results; listen to the [audio samples](benchmarks/audio_samples/).

Use `unitts benchmark --engine <name>` for a basic run. For multilingual and cloning checks, run [`benchmarks/gpu_smoke_test.py`](benchmarks/gpu_smoke_test.py) with `--engine moss-tts` or `--engine higgs-tts` and `--output-dir <directory>` on a CUDA host; start the Higgs server first.

## Adding an engine

1. Add `src/unitts/engines/<name>_engine.py`
2. Subclass `TTSEngine`, implement `load_model()` and `synthesize()`, decorate with `@register_engine`
3. If the adapter needs new packages, add them to `pyproject.toml` extras; adapters using existing dependencies need no extra
4. Add a unit test under `tests/unit/`

## License

UNITTS is Apache 2.0 (see [LICENSE](LICENSE)). Each model keeps its upstream license, listed in the Engines table. See [NOTICE](NOTICE) for third-party notices and usage terms.

**Built with Fish Audio.** Commercial use of Fish Audio s2-pro requires a separate license from Fish Audio.

**Built with Higgs TTS 3 licensed from Boson AI USA, Inc.** Its [license](https://huggingface.co/bosonai/higgs-tts-3-4b/blob/main/LICENSE) permits research, non-commercial use and attributed creator content. Hosted APIs and product/service integration require a separate commercial license.
