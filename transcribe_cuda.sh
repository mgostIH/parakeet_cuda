#!/usr/bin/env bash
set -euo pipefail
script_path="$(readlink -f -- "${BASH_SOURCE[0]}")"
root_dir="$(cd -- "$(dirname -- "$script_path")" && pwd)"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONPATH="$root_dir${PYTHONPATH:+:$PYTHONPATH}"
exec uv run --no-sync --project "$root_dir" \
  --cache-dir "${XDG_CACHE_HOME:-$HOME/.cache}/uv" python -m parakeet_cuda "$@"
