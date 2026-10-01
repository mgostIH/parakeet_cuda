> Historical local measurements. Recordings and raw artifacts are not distributed; see [public benchmark methodology](../docs/BENCHMARKS.md).

# Parakeet CUDA runtime results

The custom runtime is implemented for this host's GTX 750 Ti using CUDA 12,
ordinary FP32 cuBLAS, and handwritten Maxwell kernels. It uses the existing
Q8 GGUF. There were no additional model or framework downloads.

`transcribe_cuda.sh` is the entry point. CPU FFT and mel normalization feed
a tiled CUDA stem, all 24 CUDA FastConformer blocks, the CUDA encoder
projection, and the CUDA LSTM predictor and TDT joint decoder. Tokenization
and the greedy control loop remain on the CPU. A NumPy CPU decoder is also
available for validation and placement comparisons.

## Device memory

Four arenas are allocated once and reused throughout a recording:

| Arena | MiB |
| --- | ---: |
| Expanded current block weights | 100 |
| Compact current block weights | 28 |
| Activations and shared scratch | 96 |
| Explicit cuBLAS workspace | 16 |
| Total application reservations | 240 |

The GGUF remains in a read-only CPU mapping. No full-model FP32 expansion
is stored. A block occupies about 25.6 MiB in compact form and 96.1 MiB
after expansion. One block is uploaded at a time. Uploads are synchronous
in this implementation; no hidden prefetch/cache allocations exist.

For short inputs, encoder activations remain on device. Inputs exceeding
512 encoder frames use a layer-major host-spill schedule. Relative-position
attention visits all keys through bounded tiles with online softmax; it
does not allocate a score matrix proportional to the squared input length.
Convolutions preserve global padding and halos. The stem avoids whole-input
im2col and convolution-output buffers.

A corrected external sampler observed 280 MiB for the short CUDA process,
including context/library/module overhead. Global free-memory differences
are unsuitable for attributing memory to this runtime because the desktop
shares the GPU. The first external sampler missed uv's Python child and its
GPU/RSS readings must not be used; corrected measurements are identified
explicitly below.

## Speech test latency

The MP3 decodes to 564,160 PCM samples: 35.26 seconds, 3527 mel frames and
441 encoder frames. The decoded float32 PCM SHA-256 is
`290662db7b23c8bd9b00a1e5de3a26841740df52cb534374c6a110f18a67ff5f`.

Three CPU/CUDA pairs alternated their execution order. Both commands read
the same MP3 and include process startup, ffmpeg conversion, model access,
inference and output writing. The CPU command disables discarded warmup.
Model pages and compiled CUDA kernels were already cached. Desktop load
and GPU clock changes can affect individual results.

| Runtime | Process wall seconds | Median seconds |
| --- | --- | ---: |
| NVIDIA CPU release v0.1.0 | 15.575, 14.051, 14.303 | 14.303 |
| Custom CUDA runtime | 1.968, 1.902, 1.946 | 1.946 |

The median ratio is 7.35. Every transcript matched exactly.
Raw evidence: `artifacts/process-benchmark/results.json`.

The earlier first CUDA run took 3.348 seconds internally and the standalone
CPU WAV run took 13.95 seconds. These preliminary timings have been
superseded by the process comparison above.

## Correctness evidence

Eight numerical and host contracts cover scalar Q8 decoding, GEMV,
LayerNorm, stem seams, a complete FastConformer block, resident versus
spilled attention, TDT argmax tie handling, predictor state commitment,
zero-duration guards, segment coverage and overlap reconciliation.

Independent references use NumPy and directly decoded GGUF values. The
attention reference reproduces the upstream relative shift through
pad/reshape/drop/reshape operations, independently of the CUDA indexing.
Q8 decoding was exact. The complete 13-frame block's maximum error was
about 0.000063. Stem checks use multiple tile sizes and boundaries.

For the complete 35-second clip, forcing activation spilling changed the
final encoder values by at most 0.00000564. Token IDs and the entire greedy
token/duration trace were identical. CPU and CUDA decoder implementations
also produced identical token IDs and traces. JFK produced the expected
transcript through the complete CUDA path.

A synthetic 141.04-second recording made from four copies of the existing
clip was tested through full-context spilling and segmentation. Both used
the same fixed 240 MiB reservation. Full-context CUDA and the CPU reference
had identical lexical words, with two comma differences. CPU Q8 arithmetic
and FP32 multiplication of dequantized weights are not bitwise equivalent.
The segmented output matched four copies of the short transcript after
case/punctuation normalization. A duplicated overlap word found during the
first segmented run was fixed and covered by a host contract.

Evidence is under `artifacts/cuda_contract.json`, `artifacts/cuda_first.json`,
`artifacts/cuda_spill.json`, `artifacts/cuda_cpu_decoder.json`,
`artifacts/cuda_jfk.json`, `artifacts/cuda_long_full.json`,
`artifacts/cpu_long_full.json`, and `artifacts/cuda_segmented_fixed.json`.

## Dense kernel experiment

The direct Q8 GEMM candidate uses a 32 by 64 output tile, a reduction tile
of 32, shared-memory reuse and a 2 by 4 FP32 register tile per thread.
It passed the TIRx fixed contract, including independent float64 goldens,
real projection dimensions, tails, repeated execution and allocation guards.

Paired CUDA graph timing includes ten repeated operations per replay and
eight retained alternating samples. The cuBLAS control includes expansion
on every operation, even though the selected runtime expands each block
once and reuses the matrices across time tiles.

| Shape M K N | Direct Q8 microseconds | Expansion and SGEMM microseconds |
| --- | ---: | ---: |
| 1 32 3 | 5.216 | 4.416 |
| 17 640 65 | 104.926 | 21.688 |
| 33 1024 129 | 194.363 | 40.280 |
| 441 1024 4096 | 14971.336 | 4701.824 |

cuBLAS remains selected. These are warmed kernel/control timings, separate
from application latency. Evidence: `artifacts/q8-gemm-check` and
`artifacts/dense-comparison/results.json`.

## Long recordings

Auto mode processes recordings longer than 45 seconds as bounded windows,
choosing quiet boundaries and retaining a two-second overlap. Word ownership
and matching text/timestamps reconcile duplicates. The WAV is read a window
at a time; GPU allocations are reused between all windows. MP3 conversion
uses a temporary WAV which is removed on success or failure.

Full mode retains whole-utterance context and supports the stored positional
limit of 5000 encoder frames, approximately 400 seconds. It bounds VRAM but
retains quadratic attention work. Segmentation changes context, normalization
and potentially punctuation or recognition near boundaries. Its overlap
logic has been tested locally, not evaluated on a large labeled corpus.

The test `long-recording.wav` is 1701.5466875 seconds, or 28 minutes
21.55 seconds. Its first CUDA run completed 45 segments in 85.270 process
wall seconds, approximately 19.95 times real time. Application reservations
stayed at 240 MiB and four physical allocations for every segment. The
largest activation/scratch use was 61.973 MiB. No CPU comparison was run on
this recording. Its transcript is saved under `artifacts/long-recording-cuda`.

The first long run's external memory sampler missed the Python child;
only its latency, transcript and internal allocation ledger are valid.
The repeat with corrected process accounting is saved under
`artifacts/long-recording-cuda-verified-memory`.

The verified repeat completed in 90.075 process wall seconds, or 18.89 times
real time, and produced the identical 5209-word transcript. Total process
VRAM peaked at 280 MiB; active samples were 279 or 280 MiB throughout the
recording. The sampled process tree peaked at 913.633 MiB of host RSS,
including uv; the Python runtime's own high-water RSS was 890.297 MiB.
Every segment reported exactly four physical device allocations and 240 MiB
of reserved application buffers. Segment ownership covered 0 through
1701.5466875 seconds continuously.

Reproduce the runtime command:

```bash
./transcribe_cuda.sh long-recording.wav --progress --output artifacts/video-transcript.json
```

Reproduce the external measurement with a fresh output directory:

```bash
python3 analysis/benchmark_recording.py long-recording.wav \
  --output artifacts/video-measurement-new
```

The two full-recording runs took 85.3 and 90.1 seconds. The second establishes
the observed process VRAM bound; its wall time also includes external
sampling and polling. The final eight-test correctness gate passed after
both recording runs. No CPU inference was performed on this video.

## Further optimization opportunities

The working runtime satisfies the low-memory CUDA inference and long-input
requirements. Additional experiments can add pinned double-buffered weight
prefetch, optional compact weight caching, recurring tile graphs, dense
algorithm tuning and different attention tile sizes. None are required for
the measured speedup or memory bound. Each should beat the current complete
runtime and preserve the fixed numerical contracts before selection.
