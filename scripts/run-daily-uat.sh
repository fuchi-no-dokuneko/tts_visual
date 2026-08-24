#!/bin/sh
set -eu

python_bin=${TTS_VISUAL_PYTHON:-.venv/bin/python}
if [ ! -x "$python_bin" ]; then
    python_bin=python3
fi

ACCEPTANCE_SUITE=daily "$python_bin" -m behave --no-capture --no-capture-stderr features/daily_uat.feature
test -s uat-artifacts/checklist.json
test -s uat-artifacts/sonar-test-execution.xml

if [ "${RUN_SONAR_SCANNER:-0}" = "1" ]; then
    command -v sonar-scanner >/dev/null
    sonar-scanner -Dsonar.testExecutionReportPaths=uat-artifacts/sonar-test-execution.xml
fi
