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

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/getting-started/installation/). Choose an engine from the table below; one command installs UNITTS and its dependencies in a dedicated Python 3.12 environment.

```bash
git clone https://github.com/aynursusuz/unitts.git
cd unitts
python3 scripts/setup.py --engine moss-tts  # change the engine name to select a model
source .venv-moss-tts/bin/activate
```

To switch models, rerun the same command with another engine name and activate the environment printed by setup. Setup checks prerequisites and reports missing packages or system libraries with repair instructions. Weights download on first use.

### Before you start

Some engines need system packages or model access that setup cannot install for you:

- **eSpeak NG** (Kokoro, NeuTTS). macOS: `brew install espeak-ng`; the copies bundled with these Python packages do not load on macOS, so UNITTS uses Homebrew's. Debian/Ubuntu: `sudo apt-get install espeak-ng` if the first run reports a phonemizer error.
- **FFmpeg** (MOSS-TTS, Echo-TTS, Higgs): `brew install ffmpeg` or `sudo apt-get install ffmpeg`.
- **PortAudio** (Fish Audio on Linux): `sudo apt-get install portaudio19-dev`.
- **Linux with an NVIDIA GPU** (Echo-TTS, Higgs). Higgs needs a driver that reports CUDA 13.0 or newer in `nvidia-smi`.
- **Hugging Face access** (NeuTTS): accept the terms on [neutts-air](https://huggingface.co/neuphonic/neutts-air) and [neucodec](https://huggingface.co/neuphonic/neucodec), then run `hf auth login` in the NeuTTS environment.

For `higgs-tts`, the same setup command also installs its server. After activating `.venv-higgs-tts`, run `unitts serve --engine higgs-tts` in another terminal before inference. Stop the server with Ctrl+C when finished.

## Inference

```python
from unitts.engines import get_engine

engine = get_engine("moss-tts")  # use the engine installed above; names are in the table
try:
    engine.synthesize_to_file("Hello world!", "out.wav")
finally:
    engine.unload_model()
```

Use `get_engine("moss-tts", language="tr")` for Turkish. Higgs detects the language automatically. For MOSS or Higgs voice cloning, pass `ref_audio="ref.wav"` to `synthesize_to_file`; Higgs also accepts `ref_text`. NeuTTS requires both reference audio and its transcript. Model-specific options are documented in each [adapter](src/unitts/engines/).

### CLI

```bash
unitts list-engines
unitts synthesize "Hello world!" --engine moss-tts --output out.wav
unitts benchmark --engine moss-tts
```

`benchmark` writes one JSON to `benchmarks/results/<engine>.json` and one WAV to `benchmarks/audio_samples/<engine>.wav`. Use the Python API for language selection, reference audio and generation options.

## Engines

| Engine | Name for setup / Python | Type | Voice cloning | License | Status |
|--------|-------------------------|------|:-------------:|---------|--------|
| [Chatterbox](https://github.com/resemble-ai/chatterbox) | `chatterbox` | local | yes | MIT | integrated |
| [MOSS-TTS v1.5](https://github.com/OpenMOSS/MOSS-TTS) | `moss-tts` | local | yes | Apache-2.0 | integrated |
| [Higgs TTS 3 4B](https://huggingface.co/bosonai/higgs-tts-3-4b) | `higgs-tts` | self-hosted HTTP | yes | Boson Research and Non-Commercial | integrated |
| [Fish Audio s2-pro](https://huggingface.co/fishaudio/s2-pro) | `fish-audio` | local | yes | Fish Audio Research License | integrated |
| [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) | `qwen3-tts` | local | yes | Apache-2.0 | integrated |
| [Echo-TTS](https://github.com/FoxEngine-ai/echo-tts) | `echo-tts` | local | yes | CC-BY-NC-SA-4.0 (weights) | integrated |
| [Kokoro](https://github.com/hexgrad/kokoro) | `kokoro` | local | no | Apache-2.0 | integrated |
| [Supertonic](https://github.com/supertone-inc/supertonic) | `supertonic` | local | no | OpenRAIL-M (weights) | integrated |
| [NeuTTS](https://github.com/neuphonic/neutts) | `neutts` | local | yes | Apache-2.0 | integrated |

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

VRAM values are allocated-memory snapshots; `*` marks peak allocation for the MOSS sample. `—` means server VRAM was not measured. Row links contain full results.

Use `unitts benchmark --engine <name>` for a basic run. For multilingual and cloning checks, run [`benchmarks/gpu_smoke_test.py`](benchmarks/gpu_smoke_test.py) with `--engine moss-tts` or `--engine higgs-tts` and `--output-dir <directory>` on a CUDA host; start the Higgs server first and pass `--base-url http://127.0.0.1:8000`.

## Audio samples

All nine engines reading the same sentence: "One Python interface for every text to speech model." These are separate from the benchmark runs above, which used a longer text.

| Engine | Sample | How it was generated |
|--------|--------|----------------------|
| Kokoro | [kokoro.wav](benchmarks/audio_samples/kokoro.wav) | default voice (`af_heart`) |
| Chatterbox | [chatterbox.wav](benchmarks/audio_samples/chatterbox.wav) | no reference audio |
| Qwen3-TTS | [qwen3-tts.wav](benchmarks/audio_samples/qwen3-tts.wav) | default speaker (`Ryan`), `language="English"`, `instruct="Speak in a calm, natural, conversational tone."` |
| Supertonic | [supertonic.wav](benchmarks/audio_samples/supertonic.wav) | default voice (`M1`), `lang="en"` |
| Fish Audio s2-pro | [fish-audio.wav](benchmarks/audio_samples/fish-audio.wav) | no reference audio |
| Echo-TTS | [echo-tts.wav](benchmarks/audio_samples/echo-tts.wav) | no reference audio |
| NeuTTS | [neutts.wav](benchmarks/audio_samples/neutts.wav) | cloned from a Kokoro clip (NeuTTS needs a reference) |
| MOSS-TTS v1.5 | [moss-tts.wav](benchmarks/audio_samples/moss-tts.wav) | no reference audio, `language="en"` |
| Higgs TTS 3 4B | [higgs-tts.wav](benchmarks/audio_samples/higgs-tts.wav) | no reference audio |

## Adding an engine

1. Add `src/unitts/engines/<name>_engine.py`
2. Subclass `TTSEngine`, implement `load_model()` and `synthesize()`, decorate with `@register_engine`
3. Add required packages to `pyproject.toml` extras and a setup recipe to `scripts/setup.py`
4. Add a unit test under `tests/unit/`

## License

UNITTS is Apache 2.0 (see [LICENSE](LICENSE)). Each model keeps its upstream license, listed in the Engines table. See [NOTICE](NOTICE) for third-party notices and usage terms.

**Built with Fish Audio.** Commercial use of Fish Audio s2-pro requires a separate license from Fish Audio.

**Built with Higgs TTS 3 licensed from Boson AI USA, Inc.** Its [license](https://huggingface.co/bosonai/higgs-tts-3-4b/blob/main/LICENSE) permits research, non-commercial use and attributed creator content. Hosted APIs and product/service integration require a separate commercial license.
