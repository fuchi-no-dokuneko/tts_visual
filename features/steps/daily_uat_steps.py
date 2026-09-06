import json
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from html.parser import HTMLParser
from pathlib import Path

from behave import given, then, when


REPOSITORY = Path(__file__).resolve().parents[2]


class ReportDocument(HTMLParser):
    def __init__(self):
        super().__init__()
        self.audio_sources = []
        self.headings = []
        self.header_cells = []
        self.text_parts = []
        self._capture = None

    def handle_starttag(self, tag, attributes):
        values = dict(attributes)
        if tag == "audio" and values.get("src"):
            self.audio_sources.append(values["src"])
        self._capture = tag if tag in {"h1", "th"} else None

    def handle_endtag(self, tag):
        if self._capture == tag:
            self._capture = None

    def handle_data(self, data):
        text = data.strip()
        if not text:
            return
        self.text_parts.append(text)
        if self._capture == "h1":
            self.headings.append(text)
        elif self._capture == "th":
            self.header_cells.append(text)


def required_directory(name):
    value = os.environ.get(name)
    assert value, f"{name} is required for real-model UAT"
    path = Path(value).expanduser().resolve()
    assert path.is_dir(), f"{name} is not a directory: {path}"
    return path


def evaluator_command(context, preflight=False, existing_output="overwrite", output=None):
    command = [
        sys.executable,
        str(REPOSITORY / "batch_runner.py"),
        "--reference-root", str(context.reference_root),
        "--output", str(output or context.output),
        "--gpt-sovits-root", str(context.model_root),
        "--version", context.version,
        "--target-text", context.target_text,
        "--device", context.device,
        "--precision", context.precision,
        "--max-items", "1",
        "--existing-output", existing_output,
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


def configure_inputs(context):
    context.model_root = required_directory("TTS_VISUAL_UAT_MODEL_ROOT")
    context.reference_root = required_directory("TTS_VISUAL_UAT_REFERENCE_ROOT")
    context.version = os.environ.get("TTS_VISUAL_UAT_VERSION") or "v2"
    context.target_text = os.environ.get("TTS_VISUAL_UAT_TARGET") or "Daily evaluator acceptance sample."
    context.device = os.environ.get("TTS_VISUAL_UAT_DEVICE") or "cuda"
    context.precision = os.environ.get("TTS_VISUAL_UAT_PRECISION") or "float16"
    context.weights_file = os.environ.get("TTS_VISUAL_UAT_WEIGHTS_FILE") or None
    context.output = context.artifact_dir / "portable-report"


def ensure_completed_report(context):
    configure_inputs(context)
    session = context.uat_session
    manifest_path = context.output / "results_meta.json"
    report_path = context.output / "index.html"
    if session.evaluated and manifest_path.is_file() and report_path.is_file():
        return
    completed = execute(evaluator_command(context))
    assert completed.returncode == 0, completed.stderr
    session.evaluated = True
    session.output = context.output


def parse_report(path):
    document = ReportDocument()
    document.feed(Path(path).read_text(encoding="utf-8"))
    return document


def chromium_binary():
    return os.environ.get("TTS_VISUAL_CHROMIUM") or shutil.which("chromium") or shutil.which("chromium-browser")


def capture_report(report, screenshot, profile):
    chromium = chromium_binary()
    assert chromium, "Chromium is required for report UAT"
    completed = subprocess.run(
        [
            chromium,
            "--headless=new",
            "--no-sandbox",
            "--disable-gpu",
            f"--user-data-dir={profile}",
            f"--screenshot={screenshot}",
            "--window-size=1440,900",
            Path(report).as_uri(),
        ],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=60, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert Path(screenshot).is_file() and Path(screenshot).stat().st_size > 0


@given("real GPT-SoVITS model and reference inputs are configured")
def configure_real_inputs(context):
    configure_inputs(context)


@when("I run the evaluator preflight")
def run_preflight(context):
    context.preflight = execute(evaluator_command(context, preflight=True))


@then("preflight succeeds without loading the model")
def preflight_passes(context):
    assert context.preflight.returncode == 0, context.preflight.stderr
    assert "Preflight passed" in context.preflight.stderr


@when("I request float16 CPU execution in preflight")
def unsafe_cpu_precision(context):
    command = evaluator_command(context, preflight=True)
    command[command.index("--device") + 1] = "cpu"
    command[command.index("--precision") + 1] = "float16"
    context.unsafe_preflight = execute(command)


@then("preflight rejects the unsafe precision with a clear diagnostic")
def unsafe_preflight_rejected(context):
    assert context.unsafe_preflight.returncode != 0
    assert "CPU execution supports only --precision float32" in context.unsafe_preflight.stderr


@when("I synthesize one bounded sample with the real model")
def synthesize_sample(context):
    ensure_completed_report(context)


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


@given("a completed production comparison report is available")
def completed_report(context):
    ensure_completed_report(context)


@when("I inspect the report document")
def inspect_report_document(context):
    context.report_html = (context.output / "index.html").read_text(encoding="utf-8")
    context.report_document = parse_report(context.output / "index.html")
    context.report_manifest = json.loads((context.output / "results_meta.json").read_text(encoding="utf-8"))


@then("the report shows its title, case count, model column, subject text, and audio controls")
def report_visible_content(context):
    from utils.audio_handler import ReferenceHandler

    document = context.report_document
    assert document.headings == ["TTS Model Comparison"]
    assert "1 test cases across 1 models." in " ".join(document.text_parts)
    assert document.header_cells[:2] == ["Character / Text", "Reference"]
    assert context.version in document.header_cells
    reference_text = ReferenceHandler(str(context.reference_root)).scan_references()[0]["text"]
    assert reference_text in document.text_parts
    assert len(document.audio_sources) == 2
    assert all((context.output / source).is_file() for source in document.audio_sources)


@then("the report contains no absolute private input or output paths")
def no_private_paths(context):
    for private in (context.reference_root, context.model_root, context.output, Path.home()):
        assert str(private) not in context.report_html


@when("I copy the portable report to a different directory")
def copy_portable_report(context):
    context.moved = context.artifact_dir / "moved-portable-report"
    shutil.rmtree(context.moved, ignore_errors=True)
    shutil.copytree(context.output, context.moved)


@then("Chromium opens and reloads the moved report with local audio available")
def moved_report_opens(context):
    document = parse_report(context.moved / "index.html")
    assert document.audio_sources
    assert all(not Path(source).is_absolute() and (context.moved / source).is_file() for source in document.audio_sources)
    with tempfile.TemporaryDirectory(prefix="browser-", dir=REPOSITORY) as temporary:
        profile = Path(temporary) / "profile"
        capture_report(context.moved / "index.html", context.artifact_dir / "report-first-load.png", profile)
        capture_report(context.moved / "index.html", context.artifact_dir / "report-reload.png", profile)


def first_report_row(context):
    from utils.audio_handler import ReferenceHandler

    reference = ReferenceHandler(str(context.reference_root)).scan_references()[0]
    manifest = json.loads((context.output / "results_meta.json").read_text(encoding="utf-8"))
    item = next(item for item in manifest["items"] if item["status"] == "complete")
    row = dict(reference)
    row[context.version] = str(context.output / item["path"])
    return row


@when("I render a report that omits private reference audio")
def render_omitted_report(context):
    from utils.report_gen import ReportGenerator

    context.omit_report = context.artifact_dir / "report-reference-omitted"
    context.omit_report.mkdir(parents=True, exist_ok=True)
    ReportGenerator(str(context.omit_report)).generate_html(
        [first_report_row(context)], [context.version], reference_policy="omit",
        private_roots=(context.reference_root, context.output),
    )


@then("the report labels the reference as Omitted and keeps generated audio")
def omitted_state(context):
    document = parse_report(context.omit_report / "index.html")
    assert "Omitted" in document.text_parts
    assert len(document.audio_sources) == 1
    assert (context.omit_report / document.audio_sources[0]).is_file()


@when("I render a report with one missing model result")
def render_missing_report(context):
    from utils.report_gen import ReportGenerator

    context.missing_report = context.artifact_dir / "report-result-missing"
    context.missing_report.mkdir(parents=True, exist_ok=True)
    row = first_report_row(context)
    row.pop(context.version)
    ReportGenerator(str(context.missing_report)).generate_html(
        [row], [context.version], reference_policy="copy",
        private_roots=(context.reference_root, context.output),
    )


@then("the report labels that model result as Missing")
def missing_state(context):
    document = parse_report(context.missing_report / "index.html")
    assert "Missing" in document.text_parts
    assert len(document.audio_sources) == 1


@when("I rerun against that output with the fail policy")
def refuse_existing_output(context):
    context.manifest_before_fail = (context.output / "results_meta.json").read_bytes()
    context.fail_policy = execute(evaluator_command(context, existing_output="fail"))


@then("the evaluator refuses replacement and preserves the completed manifest")
def fail_policy_preserves(context):
    assert context.fail_policy.returncode != 0
    assert "Output already has a run manifest" in context.fail_policy.stderr
    assert (context.output / "results_meta.json").read_bytes() == context.manifest_before_fail


@then("Chromium opens the moved portable report and records a screenshot")
def open_report(context):
    copy_portable_report(context)
    moved_report_opens(context)
