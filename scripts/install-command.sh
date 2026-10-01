#!/usr/bin/env bash
set -euo pipefail
root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
command_dir="${1:-${HOME}/.local/bin}"
mkdir -p "$command_dir"
target="$command_dir/parakeet"
if [[ -e "$target" || -L "$target" ]]; then
  if [[ "$(readlink -f -- "$target")" == "$root_dir/transcribe_cuda.sh" ]]; then
    printf 'Already installed: %s\n' "$target"
    exit 0
  fi
  printf 'A different command already exists: %s\nChoose another directory.\n' "$target" >&2
  exit 1
fi
ln -s "$root_dir/transcribe_cuda.sh" "$target"
printf 'Installed: %s\nAdd this directory to PATH if needed.\nFish: fish_add_path "%s"\n' "$target" "$command_dir"
