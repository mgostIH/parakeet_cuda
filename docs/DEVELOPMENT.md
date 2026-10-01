# Development

## Normal setup

Use `./scripts/setup.sh` and the project environment. Production is independent
of TIRx; it compiles the packaged CUDA source through
`parakeet_cuda/compiler.py` and launches through the CUDA Driver API.

```bash
uv run --no-sync parakeet --help
uv run --no-sync python -m unittest discover -s tests -v
```

Missing GPU/model prerequisites result in explicit skips in host CI. To run
the GPU contracts, download both models and set `PARAKEET_RUN_GPU_TESTS=1`.
Tests do not infer accuracy from agreement between two GPU implementations:
NumPy computes independent goldens, including float64 accumulation.

## Optional unchanged C++ diarization oracle

This fetches only pinned reference source archives, without model weights:

```bash
python3 scripts/fetch_reference.py
cmake -S analysis/diar_oracle -B artifacts/diar-oracle-build -G Ninja
cmake --build artifacts/diar-oracle-build -j 4
OMP_NUM_THREADS=4 artifacts/diar-oracle-build/bin/diar_oracle \
  models/Nemotron-3-Diarization.q8_0.gguf audio.wav artifacts/reference stream
```

Requires CMake>=3.26, a C++17 compiler and Ninja. The oracle shares unchanged
upstream frontend/model/state implementations, with a small local entry point.
It accepts `stream`, `full` (short inputs), or `chunk` (raw float32 mel file).
Native probabilities are saved to `<prefix>.probs`. The build also provides
a C++ AOSC adapter; its state-equivalence test runs automatically when present.
Downloads/build products remain excluded from releases.

## Optional TIRx experiments

Historical experiments used a sibling `TIRx` workspace with `txir_tools`.
That workspace is not needed to install or run this project. If available,
its locked environment can still run the independent guarded contracts:

```bash
../TIRx/uv.sh run --locked txir check workloads/diar_rope.py
../TIRx/uv.sh run --locked txir check workloads/q8_gemm.py
```

The direct Q8 GEMM is an experiment: expansion plus cuBLAS was faster on
this GPU and remains selected. Preserve independent workload inputs and
NumPy goldens when comparing a candidate; retain compiler/source hashes,
raw samples, target GPU and workload limits with any performance claim.

## Packaging

```bash
uv build
python3 scripts/check_release.py dist/*.whl dist/*.tar.gz
```

The package includes CUDA source and Python modules. It excludes weights,
recordings, transcripts, generated HTML and compiled CUDA/third-party binaries.
The release checker enforces those exclusions. A wheel installed outside the
checkout reads weights from the user data directory or `PARAKEET_MODEL_DIR`.
Default user locations avoid writing into `site-packages`.
