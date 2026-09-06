# tts_visual

`tts_visual` compares bounded GPT-SoVITS generations and emits a movable HTML report bundle. It validates configuration before loading a model, records every item in JSON, and exits nonzero when any required item fails.

## Install

Python 3.11 or newer and `libsndfile` are required. Do not run `installBuildTool.sh` unless system packages are intentionally being provisioned.

```sh
./install.sh requirements-runtime.lock
```

GPT-SoVITS and its model files remain external. `--gpt-sovits-root` must point to a checkout containing the `GPT_SoVITS` Python package. Override the default weight paths with a JSON `--weights-file` when needed.

## Run

```sh
.venv/bin/python batch_runner.py \
  --reference-root /data/references \
  --output ./evaluation-output \
  --gpt-sovits-root /opt/GPT-SoVITS \
  --version v2 \
  --target-file ./prompt.txt \
  --device cuda \
  --precision float16 \
  --seed 7 \
  --max-items 10 \
  --max-target-chars 1000 \
  --preflight
```

Remove `--preflight` to synthesize. CPU requires `--precision float32`. Float audio is rejected when non-finite; finite values outside `[-1, 1]` are clipped by default before PCM-16 scaling, or rejected with `--audio-policy reject`.

`--existing-output resume` reuses only checksum-valid, decodable audio with the same model and parameter fingerprint. Invalid artifacts move to `quarantine/`. Use `overwrite` to regenerate or `fail` to reject an existing run. `--reference-policy copy` makes a self-contained bundle; `omit` excludes private reference audio.

## Verify

```sh
./install.sh requirements-test.lock
.venv/bin/python -m coverage run -m pytest --junitxml=test-results.xml
.venv/bin/python -m coverage report --fail-under=95
scripts/check-uat-bindings.sh
scripts/run-daily-uat.sh
scripts/run-demo-en.sh
scripts/run-demo-yue.sh
```

Real daily UAT requires `TTS_VISUAL_UAT_MODEL_ROOT` and `TTS_VISUAL_UAT_REFERENCE_ROOT`. Optional variables select version, target, device, precision, weights JSON, and Chromium. `scripts/run-daily-uat.sh` writes `checklist.json`, a screenshot, and `sonar-test-execution.xml`; set `RUN_SONAR_SCANNER=1` only where an authenticated local scanner is configured.
The complete feature matrix, visible-desktop requirement, narration timing, and TTS/recording wrapper contract are in `features/README.md`. English and Cantonese demos are recording guides, not Sonar test evidence.
