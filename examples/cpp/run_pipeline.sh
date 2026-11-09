#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 3 ]; then echo "Usage: $0 CONFIG WAV OUTPUT_DIR" >&2; exit 2; fi
work=$(mktemp -d)
trap 'rm -f -- "$work/audio.ftf"; rmdir -- "$work"' EXIT
"${AVATAR_WORKER:-avatar_worker}" extract --config "$1" --audio "$2" --output "$work/audio.ftf"
"${AVATAR_WORKER:-avatar_worker}" render --config "$1" --features "$work/audio.ftf" --output "$3"
