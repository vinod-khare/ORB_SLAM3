#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
trainer="${repo_dir}/.build/${BUILD_TYPE:-Release}/tools/vocabulary_trainer/vocabulary_trainer"
if [[ ! -x "$trainer" ]]; then
  printf 'Build the vocabulary_trainer CMake target first (%s).\n' "$trainer" >&2
  exit 1
fi
exec "$trainer" "$@"
