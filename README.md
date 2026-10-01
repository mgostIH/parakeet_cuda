# Parakeet CUDA

Speech transcription and speaker diarization with bounded GPU memory,
built around ordinary FP32 CUDA kernels and cuBLAS. Developed on a **GTX
750 Ti with 2 GiB VRAM**: measured process usage was **280 MiB for ASR** and
**384 MiB for ASR plus diarization**, including driver/library overhead. 
Speed is roughly **20x real time** for speech processing.

This is an experimental independent runtime for NVIDIA's Parakeet TDT 0.6B
v3 and Nemotron 3 Diarization GGUF models. The source is Apache-2.0.
**Model weights, recordings, transcripts and third-party binaries are not
included.** Their download links and licenses are listed below.

## What it does

- Transcribes WAV, MP3, M4A and other FFmpeg-readable audio, with word timestamps.
- Labels up to eight speakers with `--diarize`, including overlapping activity.
- Automatically creates an HTML listening report with `--diarize`: listen
  to the audio, inspect the speaker timeline and click transcript turns to seek.
- Handles long recordings through bounded audio windows and persistent
  speaker state; GPU allocation does not grow with recording length.
- Needs **NumPy, the NVIDIA driver, nvcc and cuBLAS**. Inference requires no
  PyTorch, NeMo toolkit, TVM, TIRx or CUDA Python bindings.

## Compatibility

| Component | Status |
|---|---|
| OS | Linux; other operating systems are unsupported |
| GPU | GTX 750 Ti / Maxwell `sm_50` tested |
| Other NVIDIA GPUs | Architecture detected automatically, compute capability >=5.0; untested |
| CUDA | 12.9 tested; toolkit must compile your GPU and cuBLAS must support it |
| Python | 3.12–3.13; locked NumPy 2.5.3 |
| Audio conversion | FFmpeg/ffprobe for compressed formats |

For Maxwell, Pascal and Volta, use CUDA 12.x. CUDA 13 removes compilation
and library support for those architectures. [NVIDIA release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html)
Modern GPUs may work with newer toolkits, but that path has not been
hardware-tested. All reported speed/memory numbers below are from the 750 Ti;
this is not yet a broadly validated CUDA backend.

## Quick start

Install the NVIDIA driver, a compatible CUDA toolkit including cuBLAS,
[uv](https://docs.astral.sh/uv/getting-started/installation/), and FFmpeg.
`nvidia-smi` should see your GPU. After cloning this repository:

```bash
./scripts/setup.sh
uv run --no-sync parakeet-download             # ASR GGUF only: 681 MiB
uv run --no-sync parakeet-build --arch sm_50    # GTX 750 Ti; no weights/GPU needed to compile
./transcribe_cuda.sh recording.mp3
```

`setup.sh` creates this project's own environment from `uv.lock`. It does
not require sibling repositories. For another GPU, use its architecture
with `parakeet-build`, or omit that step: first inference compiles for the
selected device automatically and caches the result.

For speaker labels and a listening report:

```bash
uv run --no-sync parakeet-download --diarize   # ASR + optional 102 MiB diarization GGUF
./transcribe_cuda.sh recording.m4a --diarize --output speakers.json
```

Existing correct models are verified and reused. These commands download
only the named GGUF files, with pinned revision/size/SHA-256 checks, and do
not download full framework checkpoints. To download just the diarizer:

```bash
uv run --no-sync parakeet-download --diarization-only
```

Non-WAV conversion happens once; its temporary WAV is removed when processing
finishes.

## HTML listening reports

Every run with `--diarize` automatically saves an HTML report. Open it in a
browser to play the audio alongside its transcript, view a colored speaker
timeline, and click a turn to jump to that part of the recording. The current
turn is highlighted during playback.

```bash
./transcribe_cuda.sh recording.m4a --diarize --output speakers.json
xdg-open speakers.html
```

This creates both `speakers.json` and `speakers.html`. Without `--output`,
the report is saved to `outputs/<audio-name>.html` in the checkout; its
absolute path is printed after processing. Plain ASR without `--diarize`
does not generate HTML.

Audio files <=10 MiB are embedded, so those reports can be shared as a single
file. Larger recordings link to the original audio using a relative path;
keep that path valid when moving the report. Browser playback depends on
support for the original audio format.

## Use from any directory

```bash
./scripts/install-command.sh
parakeet --help
parakeet "audio recording.m4a" --diarize --output "speaker transcript.json"
```

The installer adds a symlink in `~/.local/bin` without replacing a different
existing command. Ensure that directory is on PATH. In fish:

```fish
fish_add_path ~/.local/bin
```

Input/output paths use the caller's directory. The project launcher finds
its own checkout without changing directories. Ghostty needs no special
configuration. `parakeet` with no arguments displays help.

The Python package also defines `parakeet`, `parakeet-download` and
`parakeet-build` console entry points. For a wheel installed outside the
checkout, models/reports default to `$XDG_DATA_HOME/parakeet-cuda` (normally
`~/.local/share/parakeet-cuda`), and kernels use the user cache.

## Models and configuration

| Model | Download | Exact bytes | License |
|---|---|---:|---|
| [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | [Q8_0 GGUF](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/resolve/541d1f99c6b0c3cd0b11a95167540bb8edefd82b/parakeet-tdt-0.6b-v3.q8_0.gguf) | 713,975,456 | CC-BY-4.0 |
| [Nemotron 3 Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization) | [Q8_0 GGUF](https://huggingface.co/nvidia/Nemotron-3-Diarization/resolve/f667ed73aee57d40cc39428eb768b4fd87a0a29e/Nemotron-3-Diarization.q8_0.gguf) | 107,012,128 | OpenMDW-1.1 |

Hashes and revisions are defined in [download.py](parakeet_cuda/download.py).
Only these exact model architectures/schemas and Q8 layouts are supported.
Weights retain their own licenses; see [third-party notices](THIRD_PARTY_NOTICES.md).

Useful overrides:

```bash
parakeet audio.wav --device 0 --model /path/to/parakeet.gguf
parakeet audio.wav --diarize --diar-model /path/to/diarization.gguf
parakeet long.wav --progress --output transcript.json
parakeet short.wav --mode full --activation-policy spill
```

| Environment variable | Purpose |
|---|---|
| `PARAKEET_MODEL_DIR` | Shared model directory for downloader and inference |
| `PARAKEET_NVCC` | Explicit nvcc executable; useful with multiple CUDA toolkits |
| `PARAKEET_CUBLAS` | Explicit cuBLAS shared-library path |
| `PARAKEET_CACHE_DIR` | Override compiled-kernel cache directory |

The compiler cache key includes source, GPU target, compiler identity and
flags. You can build a Python wheel/source archive with `uv build`;
CUDA compilation is explicit or happens on first inference, not at package
installation. The wheel includes CUDA source, not GPU binaries.

## How memory stays bounded

The Parakeet GGUF remains in a read-only CPU mapping. One encoder block at
a time is uploaded and expanded to FP32 for cuBLAS; the full FP32 model is
never materialized. Convolution tiles have explicit halos, and long full
attention uses tiled online softmax with host activation spill.

Normal long-recording mode uses 45-second ASR windows with quiet boundaries
and two seconds of overlap. Diarization runs in a separate ordered pass on
the same converted/mapped PCM audio. Both reuse four arenas totaling 240 MiB;
the diarizer adds 104 MiB for its compact GPU weight cache. Its speaker cache
and FIFO persist across the entire recording, independent of ASR windows.
See [architecture analysis](analysis/ARCHITECTURE.md) and
[implementation results](analysis/DIARIZATION_RESULTS.md).

### Full context versus automatic windows

`--mode full` processes the entire recording as one ASR sequence, giving
each encoder frame attention over the whole recording. After the model's
8x time reduction, each encoder frame represents roughly 80 ms of audio.
The supplied model's relative-position table supports at most 5000 frames
in one sequence: approximately 400 seconds (6 minutes 40 seconds). The
runtime rejects longer input in full mode. This is a positional-table limit,
not a limit on the recording length in normal use or a VRAM capacity limit.

The default `--mode auto` processes recordings longer than 45 seconds in
overlapping windows and merges their word timestamps into one transcript.
Each window stays within the positional limit, so recordings can be much
longer than 400 seconds while GPU memory stays bounded. The tradeoff is
that ASR context is limited to each window; diarization keeps its speaker
state across the entire recording.

## Measurements

| Recording / mode | Time | Peak process VRAM |
|---|---:|---:|
| 35.26s speech, CUDA ASR; paired median |1.946s|280 MiB|
| Same speech, native CPU runtime; paired median |14.303s|—|
| 35.07s conversation, CUDA ASR + diarization; warm run |2.102s|384 MiB|
| 28m21.55s recording, CUDA ASR + diarization |103.136s|384 MiB|

The paired short-case ratio was 7.35x. Process start, audio conversion and
model access were included; CUDA compilation/model pages were warmed.
These are local sample measurements, not accuracy benchmarks or promises
for other hardware. The audio is not distributed. Reproduction commands,
methodology and caveats are in [benchmarks](docs/BENCHMARKS.md).

## Validation and development

Host checks run without GPU, weights or network:

```bash
uv run --no-sync python -m unittest discover -s tests -v
uv build
python3 scripts/check_release.py dist/*.whl dist/*.tar.gz
```

After downloading both models, run independent NumPy/CUDA contracts:

```bash
PARAKEET_RUN_GPU_TESTS=1 OPENBLAS_NUM_THREADS=1 \
  uv run --no-sync python -m unittest discover -s tests -v
```

GPU/model tests are explicitly skipped in ordinary host CI. Contracts cover
stem boundaries, complete Parakeet blocks, attention spill, greedy TDT state,
Nemotron Transformer blocks/full head, RoPE tails and host speaker-cache
behavior. The optional C++ oracle and TIRx guarded workloads are documented
in [development](docs/DEVELOPMENT.md). GitHub Actions checks host behavior
and release contents on Python 3.12/3.13 without downloading model weights.

## Known limits

- Segmented ASR changes context/normalization and can change words near boundaries.
- Explicit `--mode full` is limited to roughly 400 seconds per recording;
  default `--mode auto` handles longer recordings through windows, as
  explained [above](#full-context-versus-automatic-windows).
- Speaker labels are scoped to one recording; identity accuracy needs listening review.
- Padded speaker segments can overlap. This runtime detects overlap; it does not
  separate mixed voices or produce two simultaneous transcripts.
- Diarization tags the existing words. It does not translate or improve
  recognition of language switches. Transcribing speaker turns separately
  remains a possible follow-up.
- GPU memory is bounded; output word/segment lists and the disk-backed
  probability timeline grow with recording duration. Models also consume
  host page-cache/RSS.

See [contributing](CONTRIBUTING.md) for reproducible reports and proposed
optimizations. This repository is not an official NVIDIA project.
