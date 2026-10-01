# Benchmark methodology

These are observations on one GTX 750 Ti / 2 GiB host with CUDA 12.9, system
cuBLAS 12, Linux and NumPy 2.5.3. No other GPU has been hardware-validated.
Recordings used in measurements are not included in the repository.

## Paired speech comparison

Three CPU/CUDA process pairs alternate execution order. The same 35.26s MP3
is decoded through FFmpeg for both. Median elapsed time was 14.303s for the
NeMo-Speech.cpp v0.1.0 CPU release and 1.946s for custom CUDA:7.35x.
Compilation and model pages were warmed; CPU warm-up was disabled. Timing
includes process startup, conversion, model access and output. All transcripts
matched on this case. This does not establish universal recognition parity.

To reproduce on your own audio, obtain the optional native CPU release
from [NVIDIA](https://github.com/NVIDIA/NeMo-Speech.cpp/releases/tag/v0.1.0),
extract it under `runtime/`, then run:

```bash
uv run --no-sync python analysis/benchmark_runtime.py \
  --audio /path/to/audio.mp3 --pairs 3 --output artifacts/paired-attempt
```

The legacy CPU wrapper is only for this comparison. Production inference
does not use the packaged CPU runtime.

## Combined diarization

A 35.072s conversation took 2.1024s in the warm measured runtime, including
0.3674s diarization. A 1701.5467s recording took 103.1356s total, including
19.2990s diarization. The long recording used 45 ASR windows and 63 diarizer
chunks, with 63 speaker-cache compressions. Both measured 384 MiB total
process VRAM. No long CPU run was performed.

```bash
uv run --no-sync python analysis/benchmark_recording.py /path/to/audio.wav \
  --diarize --output artifacts/recording-attempt
```

Each attempt must have a fresh output directory. The sampler identifies
uv/Python descendants across all `/proc/PID/task/*/children` and uses
`nvidia-smi` per-process memory. Sampling/poll completion adds a small delay
to external wall time; the JSON records internal stage times too. Free-memory
differences alone are not used to attribute GPU usage on a desktop GPU.

## Correctness versus native CPU diarization

On the short conversation, the unchanged upstream C++ implementation took
3.7294s for diarization alone. Its strongest speaker matched the CUDA model
on every speech-active frame. Mean probability difference was 5.77e-5;
maximum 0.01735. CPU ggml quantizes activations for Q8 matrix products;
CUDA expands stored Q8 weights and multiplies FP32. Independent NumPy tests
validate this arithmetic separately.

Historical detailed stage/allocation reports are retained in
[ASR results](../analysis/CUDA_RESULTS.md) and
[diarization results](../analysis/DIARIZATION_RESULTS.md). References to local
artifacts describe measurement provenance; private artifacts are not included.
