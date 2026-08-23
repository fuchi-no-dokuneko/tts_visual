import json
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

from behave import given, then, when


REPOSITORY = Path(__file__).resolve().parents[2]


def required_directory(name):
    value = os.environ.get(name)
    assert value, f"{name} is required for real-model UAT"
    path = Path(value).expanduser().resolve()
    assert path.is_dir(), f"{name} is not a directory: {path}"
    return path


def evaluator_command(context, preflight=False):
    command = [
        sys.executable,
        str(REPOSITORY / "batch_runner.py"),
        "--reference-root", str(context.reference_root),
        "--output", str(context.output),
        "--gpt-sovits-root", str(context.model_root),
        "--version", context.version,
        "--target-text", context.target_text,
        "--device", context.device,
        "--precision", context.precision,
        "--max-items", "1",
        "--existing-output", "overwrite",
    ]
    if context.weights_file:
        command.extend(("--weights-file", context.weights_file))
    if preflight:
        command.append("--preflight")
    return command


def execute(command):
    return subprocess.run(
        command, cwd=REPOSITORY, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=300, check=False,
    )


@given("real GPT-SoVITS model and reference inputs are configured")
def configure_real_inputs(context):
    context.model_root = required_directory("TTS_VISUAL_UAT_MODEL_ROOT")
    context.reference_root = required_directory("TTS_VISUAL_UAT_REFERENCE_ROOT")
    context.version = os.environ.get("TTS_VISUAL_UAT_VERSION") or "v2"
    context.target_text = os.environ.get("TTS_VISUAL_UAT_TARGET") or "Daily evaluator acceptance sample."
    context.device = os.environ.get("TTS_VISUAL_UAT_DEVICE") or "cuda"
    context.precision = os.environ.get("TTS_VISUAL_UAT_PRECISION") or "float16"
    context.weights_file = os.environ.get("TTS_VISUAL_UAT_WEIGHTS_FILE") or None
    context.output = context.artifact_dir / "portable-report"


@when("I run the evaluator preflight")
def run_preflight(context):
    completed = execute(evaluator_command(context, preflight=True))
    assert completed.returncode == 0, completed.stderr
    assert "Preflight passed" in completed.stderr


@when("I synthesize one bounded sample with the real model")
def synthesize_sample(context):
    context.completed = execute(evaluator_command(context))
    assert context.completed.returncode == 0, context.completed.stderr


@then("the run manifest and generated PCM audio are valid")
def inspect_artifacts(context):
    manifest_path = context.output / "results_meta.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert len(manifest["items"]) == 1
    item = manifest["items"][0]
    assert item["status"] == "complete"
    audio_path = context.output / item["path"]
    with wave.open(str(audio_path), "rb") as audio:
        assert audio.getnframes() > 0
        assert audio.getframerate() > 0
        assert audio.getnchannels() > 0
        assert audio.getsampwidth() == 2


@then("Chromium opens the moved portable report and records a screenshot")
def open_report(context):
    moved = context.artifact_dir / "moved-portable-report"
    if moved.exists():
        shutil.rmtree(moved)
    shutil.move(context.output, moved)
    chromium = os.environ.get("TTS_VISUAL_CHROMIUM") or shutil.which("chromium") or shutil.which("chromium-browser")
    assert chromium, "Chromium is required for report UAT"
    screenshot = context.artifact_dir / "report.png"
    shutil.rmtree(context.artifact_dir / "chromium-profile", ignore_errors=True)
    with tempfile.TemporaryDirectory(prefix="browser-", dir=REPOSITORY) as temporary:
        profile_root = Path(temporary)
        profile_root.chmod(0o700)
        profile = profile_root / "profile"
        browser = subprocess.run(
            [
                chromium,
                "--headless=new",
                "--no-sandbox",
                "--disable-gpu",
                f"--user-data-dir={profile}",
                f"--screenshot={screenshot}",
                "--window-size=1440,900",
                (moved / "index.html").as_uri(),
            ],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=60, check=False,
        )
    assert browser.returncode == 0, browser.stderr
    assert screenshot.is_file() and screenshot.stat().st_size > 0
