import json
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import soundfile as sf

from tests.helpers import write_wave


REPOSITORY = Path(__file__).resolve().parents[1]


def make_model_contract(root):
    package = root / "GPT_SoVITS" / "TTS_infer_pack"
    package.mkdir(parents=True)
    (root / "GPT_SoVITS" / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (root / "torch.py").write_text(
        """class cuda:
    @staticmethod
    def is_available(): return False
    @staticmethod
    def manual_seed_all(seed): pass
    @staticmethod
    def empty_cache(): pass

def manual_seed(seed): pass
""",
        encoding="utf-8",
    )
    (package / "TTS.py").write_text(
        """import numpy as np

class TTS_Config:
    def __init__(self, values): self.values = values

class TTS:
    def __init__(self, config): self.config = config
    def run(self, inputs):
        if inputs["text"] == "FAIL":
            return iter(())
        samples = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float64)
        return iter([(8000, samples)])
""",
        encoding="utf-8",
    )
    weights = root / "GPT_SoVITS" / "pretrained_models" / "gsv-v2final-pretrained"
    weights.mkdir(parents=True)
    (weights / "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt").write_bytes(b"gpt-contract")
    (weights / "s2G2333k.pth").write_bytes(b"sovits-contract")


def run_cli(tmp_path, target="real process target", output_name="output"):
    references = tmp_path / "references"
    references.mkdir(exist_ok=True)
    write_wave(references / "real.wav")
    (references / "real.txt").write_text("real reference", encoding="utf-8")
    model = tmp_path / "model"
    if not model.exists():
        make_model_contract(model)
    output = tmp_path / output_name
    command = [
        sys.executable,
        str(REPOSITORY / "batch_runner.py"),
        "--reference-root", str(references),
        "--output", str(output),
        "--gpt-sovits-root", str(model),
        "--version", "v2",
        "--target-text", target,
        "--device", "cpu",
        "--precision", "float32",
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(model), str(REPOSITORY)))
    completed = subprocess.run(
        command, cwd=REPOSITORY, env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, check=False,
    )
    return completed, output


def test_real_process_writes_decodable_audio_json_and_html(tmp_path):
    completed, output = run_cli(tmp_path)

    assert completed.returncode == 0, completed.stderr
    manifest = json.loads((output / "results_meta.json").read_text(encoding="utf-8"))
    item = manifest["items"][0]
    audio_path = output / item["path"]
    decoded, rate = sf.read(audio_path, dtype="int16")
    with wave.open(str(audio_path), "rb") as header:
        assert header.getparams()[:4] == (1, 2, 8000, 5)
    assert rate == 8000
    assert decoded.tolist() == [-32767, -16384, 0, 16384, 32767]
    assert item["sha256"]
    assert (output / "index.html").is_file()
    assert (output / "report_manifest.json").is_file()


def test_real_process_failure_is_nonzero_and_identifies_item(tmp_path):
    completed, output = run_cli(tmp_path, target="FAIL", output_name="failed-output")

    assert completed.returncode == 1
    assert "real (v2) failed" in completed.stderr
    manifest = json.loads((output / "results_meta.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["items"][0]["name"] == "real"
    assert manifest["items"][0]["status"] == "failed"


def test_chromium_loads_moved_portable_report(tmp_path):
    completed, output = run_cli(tmp_path)
    assert completed.returncode == 0, completed.stderr
    portable_root = Path(tempfile.mkdtemp(prefix="browser-", dir=REPOSITORY))
    moved = portable_root / "moved"
    shutil.move(output, moved)
    chromium = shutil.which("chromium") or shutil.which("chromium-browser")
    assert chromium, "Chromium executable is required for ADR-258"

    try:
        browser = subprocess.run(
            [
                chromium,
                "--headless=new",
                "--no-sandbox",
                "--disable-gpu",
                f"--user-data-dir={portable_root / 'chromium-profile'}",
                "--dump-dom",
                (moved / "index.html").as_uri(),
            ],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=30, check=False,
        )

        assert browser.returncode == 0, browser.stderr
        assert "TTS Model Comparison" in browser.stdout
        assert "assets/generated/v2/sample-0001.wav" in browser.stdout
    finally:
        shutil.rmtree(portable_root, ignore_errors=True)
