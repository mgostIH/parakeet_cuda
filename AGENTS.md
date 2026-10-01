# Parakeet CUDA development

- Use jj for this local workspace; do not reinitialize its repository. A
  public clone can use its existing version-control system.
- Use the standalone `uv` project: `./scripts/setup.sh`, then
  `uv run --no-sync python -m unittest discover -s tests -v`.
- Production compilation/launch live in `parakeet_cuda/compiler.py` and
  `cuda_driver.py`. Runtime kernels are packaged under
  `parakeet_cuda/kernels/`. TIRx is optional for guarded experiments.
- GPU/model contracts require `PARAKEET_RUN_GPU_TESTS=1` and downloaded
  GGUFs. Do not fetch them or run a long CPU comparison as incidental tests.
- On this development host, GPU access needs elevated execution. Respect
  the user's existing authorization and run an actual device check.
- Keep independent NumPy/float64 goldens, fixed workloads and numerical
  tolerances when comparing kernels. GPU agreement alone is insufficient.
- Target ordinary FP32 CUDA/cuBLAS, retaining Maxwell compatibility and
  bounded allocations. New targets must remain labeled untested until
  hardware/model correctness, latency and memory are measured.
- Keep raw attempt evidence in excluded `artifacts/`, with exact sources,
  target/toolchain, warmed state and process VRAM attribution.
- Models, recordings, transcripts, generated HTML and native binaries are
  excluded. Preserve local user files; never place them in release archives.
- Use `uv build` and `scripts/check_release.py` to verify source/wheel assets.
- See docs/DEVELOPMENT.md for the optional pinned C++ oracle and TIRx checks.
