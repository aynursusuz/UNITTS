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

# Base install (includes the Higgs HTTP client; no model weights)
uv pip install -e .

# Install only the engines you need
uv pip install -e ".[chatterbox]"
uv pip install -e ".[fish-audio]"
uv pip install -e ".[qwen3-tts]"
uv pip install -e ".[echo-tts]"
uv pip install -e ".[kokoro]"
uv pip install -e ".[supertonic]"
uv pip install -e ".[neutts]"
```

> Chatterbox depends on `perth`, which still imports `pkg_resources`. On setuptools 80 or newer, also run `uv pip install "setuptools<80"`.
>
> Fish Audio pulls `fish-speech` from its GitHub repo (not on PyPI). `fish-speech` and `chatterbox-tts` currently pin different `torch` versions, so install them in separate environments. `fish-speech` also depends on `pyaudio`; on Debian/Ubuntu install its system headers first with `sudo apt-get install portaudio19-dev`.
>
> Qwen3-TTS pins `transformers==4.57.3` / `accelerate==1.12.0`, so install it in its own environment too. On CUDA you can optionally add FlashAttention 2 (`uv pip install flash-attn --no-build-isolation`) and pass `attn_implementation="flash_attention_2"` to `get_engine`.
>
> Echo-TTS needs a CUDA GPU (~8 GB VRAM). It depends on `torchcodec`, which loads the system FFmpeg libraries at runtime — install FFmpeg if it is missing. Its weights are non-commercial (CC-BY-NC-SA-4.0).

### MOSS-TTS GPU environment

Use a separate environment for MOSS-TTS: its Transformers 5.0.0 / PyTorch 2.9.1 stack conflicts with several other engines. On Linux with a CUDA 12.8-compatible driver:

```bash
uv venv --python 3.12 .venv-moss
source .venv-moss/bin/activate
uv pip install --torch-backend cu128 -e ".[moss-tts]"
# Install FFmpeg through your OS package manager if it is missing:
# sudo apt-get install ffmpeg
```

The adapter loads the official Hugging Face model code with `trust_remote_code=True`, including the checkpoint's audio tokenizer. Weights are downloaded on first use. FlashAttention is optional; the CUDA default is PyTorch SDPA. See the [upstream installation notes](https://github.com/OpenMOSS/MOSS-TTS#environment-setup).

### Higgs TTS 3 server environment

The `higgs-tts` client works with the base UNITTS installation; it needs no extra package. Run the official SGLang-Omni server in a separate environment. This pinned setup uses Python 3.12 and CUDA 13.0 wheels, so the GPU driver must support CUDA 13.0. Keep it separate from the MOSS CUDA 12.8 environment:

```bash
# From the UNITTS checkout; keep the server source outside this repository.
git clone https://github.com/sgl-project/sglang-omni.git ../sglang-omni-higgs
git -C ../sglang-omni-higgs checkout 3d4eb6e49096aa42e7f8f4e72f028823b2d85b73
uv venv --python 3.12 .venv-higgs-server
source .venv-higgs-server/bin/activate
uv pip install --prerelease=allow --torch-backend cu130 -e ../sglang-omni-higgs

sgl-omni serve \
    --model-path bosonai/higgs-tts-3-4b \
    --model-name bosonai/higgs-tts-3-4b \
    --host 127.0.0.1 --port 8001 \
    --mem-fraction-static 0.55 \
    --max-running-requests 8 \
    --cuda-graph-max-bs 8
```

In another terminal, use the UNITTS environment to call this server. Model weights download on first server startup. The client checks `/health` and verifies the configured model ID through `/v1/models`; a successful synthesis is still needed to confirm inference works. These checks do not launch the server. See the [official Higgs model card](https://huggingface.co/bosonai/higgs-tts-3-4b) and [SGLang-Omni cookbook](https://sgl-project.github.io/sglang-omni/cookbook/higgs_tts.html). Higgs GPU validation for this integration is pending.

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

`moss-tts` defaults to `OpenMOSS-Team/MOSS-TTS-v1.5`, the 8B Delay checkpoint (24 kHz mono), with 31 languages including Turkish. Pass a known language to improve multilingual synthesis:

```python
engine = get_engine("moss-tts", device="cuda", language="tr")
try:
    engine.synthesize_to_file("Merhaba! Bugün hava çok güzel.", "moss-tr.wav")
    engine.synthesize_to_file(
        "Bu ses bir referans kayıttan üretildi.", "moss-clone.wav", ref_audio="ref.wav"
    )
finally:
    engine.unload_model()  # also releases the processor's audio codec
```

`MOSS_TTS_MODEL` can also select the checkpoint. `language` accepts ISO codes or English names; `"auto"` omits the language tag. `tokens` controls target audio-frame count, and generation options such as `max_new_tokens` or `audio_temperature` are forwarded to upstream. This adapter returns complete audio; it does not expose upstream streaming.

Optional: select the 4B Local variant with `model_path="OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"`; its 48 kHz stereo output stays `[samples, channels]` in `TTSResult.audio` and the WAV.

`higgs-tts` uses `bosonai/higgs-tts-3-4b` through the server above. It supports 100+ languages including Turkish and infers the language from the text; do not pass a `language` option. Plain synthesis and reference-based cloning use the same interface:

```python
engine = get_engine("higgs-tts", base_url="http://127.0.0.1:8001")
try:
    engine.synthesize_to_file("Merhaba! Bugün nasılsınız?", "higgs-tr.wav")
    engine.synthesize_to_file(
        "Bu cümleyi referans kaydındaki sesle okuyorum.",
        "higgs-clone.wav",
        ref_audio="ref.wav",
        ref_text="Merhaba, bu bir ses örneğidir.",  # must match ref.wav exactly
        temperature=0.8,
        top_k=50,
        max_new_tokens=2048,
    )
finally:
    engine.unload_model()  # disconnects this client; the server keeps running
```

Reference audio is sent as a base64 data URI, so it can be a file on the client machine. Inline controls such as `<|emotion:contentment|>` and `<|prosody:pause|>` are preserved. The adapter returns complete WAV audio, and its timing includes the HTTP round trip and WAV decoding. It reports server VRAM as unavailable; `device=` does not change the server's GPU placement. Stop the server explicitly when finished, for example with Ctrl+C in its terminal.

Set `HIGGS_TTS_BASE_URL` to use another endpoint (the client default is `http://127.0.0.1:8000`). `HIGGS_TTS_MODEL` or `model_path=` must match the ID advertised by `/v1/models`; `HIGGS_TTS_API_KEY` or `api_key=` supplies optional bearer authentication. The base client does not start or install SGLang-Omni.

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

The CLI supports basic synthesis; use the Python API above for language selection, reference audio and model-specific generation options.

## Engines

| Engine | Type | Voice cloning | License | Status |
|--------|------|:-------------:|---------|--------|
| [Chatterbox](https://github.com/resemble-ai/chatterbox) | local | yes | MIT | integrated |
| [MOSS-TTS v1.5](https://github.com/OpenMOSS/MOSS-TTS) | local | yes | Apache-2.0 | GPU smoke test passed (8B) |
| [Higgs TTS 3 4B](https://huggingface.co/bosonai/higgs-tts-3-4b) | self-hosted HTTP | yes | Boson Research and Non-Commercial + Creator Use Grant | adapter added; GPU validation pending |
| [Fish Audio s2-pro](https://huggingface.co/fishaudio/s2-pro) | local | yes | Fish Audio Research License | integrated |
| [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) | local | yes | Apache-2.0 | integrated |
| [Echo-TTS](https://github.com/FoxEngine-ai/echo-tts) | local | yes | CC-BY-NC-SA-4.0 (weights) | integrated |
| [Kokoro](https://github.com/hexgrad/kokoro) | local | no | Apache-2.0 | integrated |
| [Supertonic](https://github.com/supertone-inc/supertonic) | local | no | OpenRAIL-M (weights) | integrated |
| [NeuTTS](https://github.com/neuphonic/neutts) | local | yes | Apache-2.0 | integrated |

## Benchmark

| Engine | RTF | Inference (s) | Audio (s) | VRAM (MB) | Sample rate |
|--------|-----|---------------|-----------|-----------|-------------|
| Chatterbox | 0.44 | 5.90 | 13.52 | 3,107 | 24,000 |
| Fish Audio s2-pro | 4.00 | 38.29 | 9.57 | 19,105 | 44,100 |
| Qwen3-TTS | 1.07 | 22.61 | 21.12 | 4,014 | 24,000 |
| Echo-TTS | **0.14** | 4.05 | 28.42 | 6,486 | 44,100 |

*RTF (real-time factor) = inference time / audio duration. Lower is faster.* Fish Audio measurements are without `--compile`; upstream documents ~5x speedup after kernel fusion. Full results: [`benchmarks/results/`](benchmarks/results/). Audio samples: [`benchmarks/audio_samples/`](benchmarks/audio_samples/). Chatterbox and Fish Audio were measured on an A100; Qwen3-TTS and Echo-TTS on an H100. Echo-TTS reaches the GPU only with a recent CUDA `torch` build; on older drivers it falls back to CPU.

These historical results are not a controlled speed ranking: GPU hardware and input texts differ. The runner records one inference and allocated VRAM snapshots, not repeated warm measurements or peak VRAM. Compare engines on the same GPU and text before making performance or cost claims.

### GPU smoke test: MOSS-TTS v1.5 8B

On October 5, 2026, `OpenMOSS-Team/MOSS-TTS-v1.5` passed Turkish, English, French and Turkish voice-cloning smoke tests on an NVIDIA RTX 6000 Ada Generation (48 GB), with PyTorch 2.9.1+cu128 and Transformers 5.0.0. All four outputs were 24 kHz mono. The cloning case used the first generated Turkish clip as its reference.

| Case | Audio (s) | Inference (s) | RTF | Peak allocated VRAM (MiB) |
|------|-----------|---------------|-----|---------------------------|
| Turkish, short text | 5.28 | 2.57 | 0.487 | 23,168 |

Full texts, all four cases and environment details: [`benchmarks/results/gpu-smoke/moss-v1.5.json`](benchmarks/results/gpu-smoke/moss-v1.5.json). [Listen to the Turkish sample](benchmarks/audio_samples/moss-tts.wav). These are individual warm samples, excluding initial model loading, and demonstrate integration rather than a quality ranking or a minimum-VRAM requirement. The peak allocation above covers the Turkish case; other inputs and cloning can use more memory. The optional Local checkpoint was not tested in this run. Higgs GPU validation remains pending.

### Run GPU smoke checks

The helper [`benchmarks/gpu_smoke_test.py`](benchmarks/gpu_smoke_test.py) performs a warmup followed by Turkish, English, French and reference-cloning cases. It checks that each WAV is finite, nonempty and nonsilent, and saves the WAVs and `result.json` in the chosen directory. It does not score pronunciation or speaker similarity.

Run from the UNITTS checkout on the CUDA host, using an environment where `unitts` is installed. This GPU-specific helper requires a local CUDA-enabled PyTorch installation even for Higgs; the ordinary Higgs HTTP client does not. Run the engines one at a time:

```bash
source .venv-moss/bin/activate
python benchmarks/gpu_smoke_test.py \
    --engine moss-tts \
    --model-path OpenMOSS-Team/MOSS-TTS-v1.5 \
    --output-dir benchmarks/audio_samples/gpu-smoke/moss-v1.5

# After the MOSS check exits, start the Higgs server in its own terminal.
# Use a separate UNITTS client environment with CUDA PyTorch for this helper.
source .venv/bin/activate
python benchmarks/gpu_smoke_test.py \
    --engine higgs-tts \
    --base-url http://127.0.0.1:8001 \
    --output-dir benchmarks/audio_samples/gpu-smoke/higgs-tts
```

MOSS reports peak allocated VRAM. Higgs reports HTTP timing and leaves server VRAM unmeasured; client PyTorch allocation would not represent the server process.

## Adding an engine

1. Add `src/unitts/engines/<name>_engine.py`
2. Subclass `TTSEngine`, implement `load_model()` and `synthesize()`, decorate with `@register_engine`
3. If the adapter needs new packages, add them to `pyproject.toml` extras; adapters using existing dependencies need no extra
4. Add a unit test under `tests/unit/`

## License

unitts itself is Apache 2.0 (see [LICENSE](LICENSE)). Each integrated model keeps its own upstream license; by invoking an engine you agree to the terms of its model. Third-party model notices are listed in [NOTICE](NOTICE).

**Built with Fish Audio.** The `fish-audio` engine uses Fish Audio s2-pro weights under the Fish Audio Research License (non-commercial). Commercial use of that engine requires a separate license from Fish Audio.

The `qwen3-tts` engine uses Qwen3-TTS weights from the Qwen team at Alibaba Cloud, released under Apache 2.0.

The `echo-tts` engine uses Echo-TTS weights (`jordand/echo-tts-base`) under CC-BY-NC-SA-4.0 (non-commercial research); the `echo-tts` code is MIT. Commercial use of the weights is not permitted.

The `kokoro` engine uses Kokoro-82M weights (`hexgrad/Kokoro-82M`), released under Apache 2.0.

The `supertonic` engine uses Supertonic weights from Supertone under the OpenRAIL-M license; the sample code is MIT.

The `neutts` engine defaults to NeuTTS-Air weights (`neuphonic/neutts-air`) from Neuphonic, released under Apache 2.0; the `neutts-nano` checkpoints use the NeuTTS Open License.

The `moss-tts` engine uses MOSS-TTS v1.5 / Local Transformer v1.5 and MOSS-Audio-Tokenizer weights from OpenMOSS / MOSI.AI under Apache 2.0.

**Built with Higgs TTS 3 licensed from Boson AI USA, Inc.** The `higgs-tts` engine uses weights under the [Boson Higgs TTS 3 Research and Non-Commercial License](https://huggingface.co/bosonai/higgs-tts-3-4b/blob/main/LICENSE), a source-available model license. Its Creator Use Grant permits creators to publish and monetize creative content, including podcasts, videos and audiobooks, with prominent attribution to Boson AI's Higgs Audio in the audio or accompanying text. That grant does not cover serving the model to third parties or embedding it in a product or service; those uses require a separate commercial license. See the full terms and upstream notices before deployment.
