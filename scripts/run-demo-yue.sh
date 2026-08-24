#!/bin/sh
set -eu

python_bin=${TTS_VISUAL_PYTHON:-.venv/bin/python}
if [ ! -x "$python_bin" ]; then
    python_bin=python3
fi

artifact_dir=${UAT_ARTIFACT_DIR:-uat-artifacts/demo-yue}
ACCEPTANCE_SUITE=demo-yue UAT_ARTIFACT_DIR="$artifact_dir" \
    "$python_bin" -m behave --no-capture --no-capture-stderr features/demo_yue.feature
test -s "$artifact_dir/checklist.json"
