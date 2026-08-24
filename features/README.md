# tts_visual acceptance and recording suites

This repository keeps its three Gherkin pipelines in the existing Behave structure:

- `daily_uat.feature` is the real-model human/AI acceptance checklist.
- `demo_en.feature` is a timed English product introduction and review guide.
- `demo_yue.feature` is the equivalent Traditional Chinese Cantonese introduction and guide.

Validate all feature syntax and step bindings without loading a model or opening Chromium:

```sh
scripts/check-uat-bindings.sh
```

Run the complete daily workflow or a recording workflow:

```sh
scripts/run-daily-uat.sh
scripts/run-demo-en.sh
scripts/run-demo-yue.sh
```

Daily execution requires `TTS_VISUAL_UAT_MODEL_ROOT`, `TTS_VISUAL_UAT_REFERENCE_ROOT`, the selected model weights, and Chromium. Optional variables select version, target, device, precision, weight mapping, and browser binary. The complete daily process includes a real GPT-SoVITS synthesis, so do not treat a preflight-only or dry run as UAT success.

The daily run writes `uat-artifacts/checklist.json`, `sonar-test-execution.xml`, and two moved-report screenshots. Demo evidence is isolated under `uat-artifacts/demo-en/` or `uat-artifacts/demo-yue/` and must not be submitted as test coverage.

## Recording contract

The recording suites require a visible `DISPLAY` or `WAYLAND_DISPLAY`. Each optional wrapper variable names one executable:

- `DEMO_TTS_COMMAND` receives `DEMO_TTS_LANGUAGE`, `DEMO_TTS_TEXT`, and `DEMO_TTS_MIN_SECONDS`.
- `DEMO_RECORD_START_COMMAND` receives `DEMO_SUITE` and `DEMO_REPOSITORY` and must return after recording starts.
- `DEMO_RECORD_STOP_COMMAND` receives the same values and must finalize the recording.

Narration always reserves the declared minimum time. With no TTS wrapper, text is printed and the same pause is retained. The operator owns the final MP4 path and publication.

## Feature coverage matrix

| User-visible behavior | Daily scenario |
| --- | --- |
| Valid preflight and unsafe CPU/float16 rejection | Validate safe configuration and reject an unsafe precision transition |
| Bounded real synthesis, complete manifest, decodable PCM-16 | Evaluate one bounded sample with the configured production model |
| Report title, counts, model/subject columns, local audio players | Inspect all visible comparison report content |
| Private-path redaction | Inspect all visible comparison report content |
| Folder portability, Chromium load/reload, local audio resolution | Move and reload the portable report |
| Omitted private reference and missing-generation labels | Show explicit privacy and missing-result states |
| Existing-output fail policy and artifact preservation | Refuse accidental replacement of existing output |

Both recording suites introduce bounded evaluation, explain row-by-row reference/generated listening, and finish with offline portability and reference-audio privacy guidance.
