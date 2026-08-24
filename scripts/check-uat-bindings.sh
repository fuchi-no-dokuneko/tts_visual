#!/bin/sh
set -eu

python_bin=${TTS_VISUAL_PYTHON:-.venv/bin/python}
if [ ! -x "$python_bin" ]; then
    python_bin=python3
fi

for feature in features/daily_uat.feature features/demo_en.feature features/demo_yue.feature; do
    "$python_bin" -m behave --dry-run --no-summary "$feature"
done
