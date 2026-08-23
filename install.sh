#!/bin/sh
set -eu

lockfile=${1:-requirements-test.lock}
environment=${TTS_VISUAL_VENV:-.venv}
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

test -f "$lockfile"
python3 -m venv "$environment"

while IFS= read -r requirement; do
    case "$requirement" in
        ""|\#*) continue ;;
    esac
    timeout 60s env PYTHONPATH="$root/tools/ipv4only" \
        PIP_INDEX_URL="${TTS_VISUAL_PIP_INDEX_URL:-https://pypi.org/simple}" \
        PIP_DISABLE_PIP_VERSION_CHECK=1 \
        "$environment/bin/python" -m pip install "$requirement"
done < "$lockfile"
