#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 AUDIO.wav|AUDIO.mp3 [nemo-speech transcribe options]" >&2
  exit 2
fi

root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runtime="$root_dir/runtime/nemo-speech-0.1.0-linux-x86_64-cpu/bin/nemo-speech"
model="$root_dir/models/parakeet-tdt-0.6b-v3.q8_0.gguf"
audio="$1"
shift

temp_wav=""
cleanup() {
  if [[ -n "$temp_wav" ]]; then
    rm -f -- "$temp_wav"
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ "${audio,,}" == *.mp3 ]]; then
  if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg is required to transcribe MP3 files" >&2
    exit 127
  fi
  temp_wav="$(mktemp "${TMPDIR:-/tmp}/parakeet-XXXXXXXX.wav")"
  ffmpeg -hide_banner -loglevel error -nostdin -y -i "$audio" \
    -ac 1 -ar 16000 -c:a pcm_s16le "$temp_wav"
  audio="$temp_wav"
fi

"$runtime" transcribe "$audio" --model "$model" --device cpu "$@"
