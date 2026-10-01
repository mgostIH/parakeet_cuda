> Historical local measurements. Recordings and raw artifacts are not distributed; see [public benchmark methodology](../docs/BENCHMARKS.md).

# Nemotron 3 integration evidence — GTX 750 Ti

Tested 2026-10-01 using CUDA 12.9 `sm_50`, the existing cuBLAS 12,
NumPy/TIRx environment and the local NVIDIA driver. No PyTorch/NeMo toolkit,
CUDA 13 or additional large speech models were installed.

## Downloads and provenance

- NVIDIA Nemotron 3 Diarization Q8 GGUF: **107,012,128 bytes / 102.0547 MiB**.
- Pinned HF revision: `f667ed73aee57d40cc39428eb768b4fd87a0a29e`.
- SHA-256: `08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1`.
- The existing NeMo-Speech.cpp source at `4c101bc7113f49101a3e11d2c994c519f41939f6`
  supplied architecture/state semantics and unchanged CPU validation code.
- Its pinned ggml source archive at `c03b4e2bcece5134827881af90242086daf75be5`
  was 3,067,538 bytes compressed. This was the only additional build dependency.
- The minimal oracle is built from the upstream runtime, frontend, Transformer,
  Sortformer and AOSC files. It needs neither SentencePiece nor the rest of
  the NeMo toolkit. It is used only for validation, not production inference.

Model source: https://huggingface.co/nvidia/Nemotron-3-Diarization

## Implementation

The production path is entirely this repository's CUDA/NumPy runtime:

1. FFmpeg converts MP3/M4A to mono 16 kHz PCM16 once; context cleanup removes
   the temporary WAV on success/failure. A single read-only host mapping
   serves both models. Current windows alone are copied to float32.
2. Existing Parakeet CUDA transcription emits the original text/timestamps.
3. Nemotron uses feature stacking (8 raw log-mel frames), projection to 512,
   input LayerNorm, 31 pre-LN RoPE Transformer blocks, final LayerNorm,
   projection to 192, Conv1D/subpixel 8x upsampling, ReLU/linear head, sigmoid.
4. Ordered GPU passes share activation/expanded-weight/cuBLAS workspaces.
   A 104 MiB compact GPU cache keeps the entire diarization GGUF, expanding
   only the current layer to FP32. No per-chunk host weight upload is needed.
5. New CUDA kernels fuse packed QKV transpose with NEOX RoPE, scale with
   softmax, bias with exact-erf GELU, and gather Conv1D im2col for cuBLAS.
   No native FP16 arithmetic, Tensor Cores or post-Maxwell instructions.
6. Host AOSC and FIFO preserve state across the whole recording. The chosen
   file-processing geometry is NVIDIA's documented 340/40 high-latency
   configuration: cache264, FIFO40, chunk340, update300, left0, right40 on the
   80 ms grid. Encoder sequence length is at most684, independent of duration.
7. Native 10 ms probabilities go into a temporary host mapping. Words use
   the upstream 160 ms onset anchor, with `unknown` for mean activity<0.3.
   Simultaneous activity >=0.5 is retained in `active_speakers`.
8. JSON includes untouched ASR text plus word labels, speaker turns and
   hysteresis/padded/merged speaker segments. Text output shows speaker turns.

The conversion is shared, while the two models compute their own mel
features: Parakeet requires per-utterance normalization and invalid-frame
masking; Nemotron requires continuous, unnormalized features. GPU execution
is sequential. This avoids resource contention and duplicate work arenas.

## Memory

| Allocation | MiB |
|---|---:|
| Expanded current weights |100|
| Raw Parakeet transfer staging |28|
| Shared scratch |96|
| cuBLAS workspace |16|
| Compact diarization model cache |104|
| **Fixed physical arenas, five allocations** |**344**|
| **Observed total process VRAM including driver/library overhead** |**384**|

Both sample and 28-minute recording measured384 MiB peak via nvidia-smi,
following every `/proc/PID/task/*/children` to identify the uv/Python process.
There is no allocation proportional to recording length on the GPU.
Diarization state after updates stays at cache<=264 and FIFO<=40; compression
has bounded temporary arrays. The output word/segment lists and disk-backed
probability timeline grow with duration. Read-only model mappings also consume
host page-cache/RSS. Approximate measured process-tree RSS was983 MiB short
and1069 MiB long; this is primarily mapped weights, not stored whole-file audio.

## Italian/Hungarian two-speaker recording

Input: `two-speakers.m4a`, container35.1s, decoded35.072s.

- First inference:3.4900s, diarization0.3948s (includes encoder first-use cost).
- Warm measured inference: **2.1024s**, including **0.3674s diarization**.
- External sampling wall:2.6417s (includes startup/poll completion delay).
- 75 words,2 detected speakers,8 displayed turns including2 `unknown` turns.
- Hungarian ending is assigned to speaker2, consistent with the user's
  description; actual speaker turn boundaries need the user's listening review.
- The existing recognizer still outputs poor Hungarian text. Diarization
  does not revise or translate words. Speaker-turn ASR is a future experiment.

Reviewable outputs:

- `outputs/two-speakers.html`: portable player/timeline,
  click a turn to seek, embeds the original small M4A.
- `outputs/two-speakers.txt`: speaker-labeled text.
- `outputs/two-speakers.json`: words/activity/turns/segments.

The unchanged upstream C++ CPU streaming implementation took3.7294s for
**diarization alone**, using the same geometry. Against its3508x8 probabilities:

- mean absolute difference:5.77e-5;
- 99th percentile absolute difference:0.0017904;
- maximum absolute difference:0.0173512;
- strongest-channel agreement on speech-active frames: **100%**;
- strongest-channel agreement over all frames:99.9715%.

CPU ggml quantizes activations for Q8 matmuls; this CUDA implementation
expands stored Q8 weights and uses FP32 SGEMM, so exact probability parity is
not expected. The independent NumPy contracts below establish CUDA arithmetic
correctness separately.

Evidence: `artifacts/ita-hu-diar-measured/measurement.json`,
`artifacts/ita-hu-cpp-comparison.json`, `artifacts/ita-hu-cpp.probs`,
`artifacts/ita-hu-diar-first.npy`.

## Long recording

Input:`long-recording.wav`,1701.5466875s /28m21.55s, previously testedASR-only.

- **103.1356s internal total;103.5709s external wall**.
- **19.2990s diarization**,17.4893s model computation,1.1244s frontend.
- 45 ASR segments;63 diarizer chunks and63 cache compressions.
- 5209 original ASR words; the recognizer output is unchanged by speaker labeling.
- Five fixed GPU allocations, **384 MiB observed peak**.
- Max speaker cache264, FIFO40.
- Model emitted activity in all8 speaker channels on this edited video;
  speaker identity accuracy on that video has not been manually evaluated.
- No CPU transcription or CPU diarization run on this long recording.

Evidence:`artifacts/long-recording-diar-measured/measurement.json` and transcript.

## Correctness gates

- All14 repository tests pass, including existing Parakeet regressions.
- CUDA QKV/NEOX RoPE tails1/13/67 and real layers0/15/30 vs independent
  float64 NumPy: worst complete-block error3.7384e-4.
- Complete31-layer network including padded feature stacking and output
  convolution/head vs independent NumPy: worst probability error2.91e-7.
- Frame-global frontend matches arbitrary internal and final slices of a
  whole-file calculation, including real pre-emphasis history.
- Word anchor, overlap metadata and onset/offset segmentation tested.
- AOSC port matches unchanged upstream C++ cache embeddings, FIFO and stored
  predictions exactly over repeated compression, clean speech, overlap,
  silence, tied scores and a partial tail.
- TIRx fixed RoPE contract passes all5 cases including684 and768 frames,
  read-only inputs, repeated execution and sentinel output guards.

Evidence:`artifacts/diar-final-tests.log`, `artifacts/diar_contract.json`,
`artifacts/diar-rope-contract/results.json`.

## Current limits

The model supports up to8 voices per recording, with labels scoped to that
recording. Cache continuity avoids reset at ASR chunk boundaries; it does not
provide a ground-truth guarantee against speaker confusion. Overlap detection
is not source separation and cannot create two independent overlapping
transcripts. Word timestamps can precede detected speech; those anchors can
be marked unknown. The streaming geometry has a fixed maximum768 frames in
this GPU implementation. Transcribing turns separately remains unimplemented
per the user's requested priority.
