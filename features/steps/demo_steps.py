import os
import shutil
import subprocess
import time
from html.parser import HTMLParser
from pathlib import Path

from behave import given, then, use_step_matcher, when


REPOSITORY = Path(__file__).resolve().parents[2]


class DemoReport(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []
        self.audio_sources = []

    def handle_starttag(self, tag, attributes):
        values = dict(attributes)
        if tag == "audio" and values.get("src"):
            self.audio_sources.append(values["src"])

    def handle_data(self, data):
        value = data.strip()
        if value:
            self.text.append(value)


def run_wrapper(variable, extra_environment):
    executable = os.environ.get(variable)
    if not executable:
        return False
    completed = subprocess.run(
        [executable], env={**os.environ, **extra_environment}, timeout=60, check=False
    )
    assert completed.returncode == 0, f"{variable} exited with status {completed.returncode}"
    return True


def stop_browser(context):
    process = getattr(context, "demo_browser_process", None)
    if process and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    context.demo_browser_process = None


@given("I begin a recorded demonstration")
@when("I begin a recorded demonstration")
def begin_recording(context):
    run_wrapper(
        "DEMO_RECORD_START_COMMAND",
        {"DEMO_SUITE": context.acceptance_suite, "DEMO_REPOSITORY": "tts_visual"},
    )
    context.demo_recording = True


@when("I open the comparison report for recording")
def open_recording_report(context):
    chromium = os.environ.get("TTS_VISUAL_CHROMIUM") or shutil.which("chromium") or shutil.which("chromium-browser")
    assert chromium, "A visible Chromium session is required for the recording demo"
    assert os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"), "A visible desktop session is required for the recording demo"
    report = context.output / "index.html"
    context.demo_report = DemoReport()
    context.demo_report.feed(report.read_text(encoding="utf-8"))
    context.demo_profile = context.artifact_dir / "demo-chromium-profile"
    shutil.rmtree(context.demo_profile, ignore_errors=True)
    context.demo_browser_process = subprocess.Popen(
        [
            chromium,
            "--no-sandbox",
            "--disable-session-crashed-bubble",
            "--no-first-run",
            f"--user-data-dir={context.demo_profile}",
            "--window-size=1440,900",
            "--new-window",
            report.as_uri(),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(3)
    assert context.demo_browser_process.poll() is None, "Chromium closed before recording began"


use_step_matcher("re")


@when(r'I narrate in "(?P<language>[^"]+)" for at least (?P<minimum_seconds>\d+) seconds')
def narrate(context, language, minimum_seconds):
    minimum_seconds = int(minimum_seconds)
    narration = (context.text or "").strip()
    assert narration, "Narration text is required"
    started = time.monotonic()
    invoked = run_wrapper(
        "DEMO_TTS_COMMAND",
        {
            "DEMO_TTS_LANGUAGE": language,
            "DEMO_TTS_TEXT": narration,
            "DEMO_TTS_MIN_SECONDS": str(minimum_seconds),
        },
    )
    if not invoked:
        print(f"NARRATION [{language}, >={minimum_seconds}s]: {narration}")
    remaining = minimum_seconds - (time.monotonic() - started)
    if remaining > 0:
        time.sleep(remaining)


use_step_matcher("parse")


@then("the recording view shows the report title and evaluated case count")
def recording_summary(context):
    text = " ".join(context.demo_report.text)
    assert "TTS Model Comparison" in text
    assert "test cases across" in text


@when("I present the reference and generated audio columns")
def present_audio_columns(context):
    assert "Reference" in context.demo_report.text
    assert context.version in context.demo_report.text
    assert len(context.demo_report.audio_sources) >= 2
    assert all((context.output / source).is_file() for source in context.demo_report.audio_sources)
    time.sleep(3)


@when("I present the portable and privacy behavior")
def present_portability(context):
    assert all(not Path(source).is_absolute() for source in context.demo_report.audio_sources)
    manifest = context.output / "report_manifest.json"
    assert manifest.is_file()
    time.sleep(3)


@then("I finish the recorded demonstration")
def finish_recording(context):
    if getattr(context, "demo_recording", False):
        run_wrapper(
            "DEMO_RECORD_STOP_COMMAND",
            {"DEMO_SUITE": context.acceptance_suite, "DEMO_REPOSITORY": "tts_visual"},
        )
        context.demo_recording = False
    stop_browser(context)
