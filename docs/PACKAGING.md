# Release preparation

The repository now builds a standalone Python wheel and source distribution.
Neither normal setup nor inference needs a sibling TIRx checkout. The public
source includes architecture notes, benchmark methodology, independent
contracts, optional research workloads and a pinned C++ reference adapter.

## Local verification (2026-10-01)

- Project environment: Python3.13.5 /NumPy2.5.3; all18 model/GPU/host checks passed.
- Source archive extracted into a fresh `/tmp` directory: Python3.12.3,
  own environment, no models or TIRx. Host checks passed; GPU/model/reference
  checks reported explicit skips.
- Wheel installed in a separate `/tmp` environment: correct console help,
  CUDA source included, user data/cache defaults, no imported TIRx/TVM.
- Installed wheel ran real ASR+diarization on the750Ti using an explicit
  external model directory. Text matched the existing result, two speakers
  were reported, automatic HTML was produced and reservations remained344MiB.
- Model downloader verified both local pinned files without redownloading.
- Source/wheel release checker found no weights, media, transcripts, generated
  HTML, native binaries or downloaded dependencies. Tracked-source inventory
  was also checked for those categories.
- The existing fish `parakeet` command still resolves to the project launcher.

GitHub Actions is configured for host/package checks on Python3.12 and3.13.
The local verification above does not include GitHub Actions results.
CUDA12.9 /sm_50 is the only GPU/toolkit
combination tested. Automatic target compilation removes the original strict
750Ti restriction, but it does not establish correctness or performance on
another physical GPU. Windows and macOS support are not implemented.

## Build the reviewed artifacts

```bash
./scripts/setup.sh
uv build
python3 scripts/check_release.py dist/*.whl dist/*.tar.gz
```

The `.tar.gz` contains the clean source tree, including `.github/workflows`,
README, Apache2.0 license/notices and build scripts. It is suitable for
starting a public repository without copying this workspace's downloaded
models, research caches or personal files. The wheel contains Python and
CUDA source; it compiles kernels for the selected device after installation.

The public repository is [mgostIH/parakeet_cuda](https://github.com/mgostIH/parakeet_cuda).
Only the reviewed source tree is published; local recordings, model weights,
the reimplementation plan and generated outputs remain excluded.
