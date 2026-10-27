#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if ! command -v clang-format >/dev/null 2>&1; then
    echo "Error: clang-format is required but was not found in PATH." >&2
    exit 1
fi

git ls-files -z --cached --others --exclude-standard -- \
    '*.cpp' '*.cc' '*.cxx' '*.c++' '*.C' \
    '*.h' '*.hh' '*.hpp' '*.hxx' '*.h++' '*.ipp' '*.tpp' '*.inl' |
    while IFS= read -r -d '' file; do
        if [[ -f "$file" && ! -L "$file" ]]; then
            clang-format -i --style=file -- "$file"
        fi
    done
