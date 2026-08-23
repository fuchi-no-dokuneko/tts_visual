import wave
from dataclasses import replace
from pathlib import Path

import numpy as np

from config import EvaluationConfig


def write_wave(path, samples=None, sample_rate=8000):
    values = samples if samples is not None else [0] * (sample_rate * 3)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(np.asarray(values, dtype="<i2").tobytes())


def make_config(tmp_path, **changes):
    reference_root = tmp_path / "references"
    reference_root.mkdir(parents=True)
    write_wave(reference_root / "voice.wav")
    (reference_root / "voice.txt").write_text("reference prompt", encoding="utf-8")

    model_root = tmp_path / "model"
    model_root.mkdir()
    entry_point = model_root / "GPT_SoVITS" / "TTS_infer_pack" / "TTS.py"
    entry_point.parent.mkdir(parents=True)
    entry_point.write_text("", encoding="utf-8")
    weights = {"v2": {"gpt": "gpt.ckpt", "sovits": "sovits.pth"}}
    (model_root / "gpt.ckpt").write_bytes(b"gpt")
    (model_root / "sovits.pth").write_bytes(b"sovits")
    config = EvaluationConfig(
        reference_root=reference_root,
        output=tmp_path / "output",
        gpt_sovits_root=model_root,
        versions=("v2",),
        weights=weights,
        target_text="target prompt",
        language="ja",
        device="cpu",
        precision="float32",
        seed=7,
        max_items=10,
        max_target_chars=1000,
        audio_policy="clip",
        existing_output="resume",
        reference_policy="copy",
        preflight_only=False,
    )
    return replace(config, **changes)


class SuccessfulEngine:
    inference_count = 0

    def __init__(self, **kwargs):
        self.arguments = kwargs

    def load_version(self, version):
        self.version = version

    def infer(self, ref_wav, ref_text, target_text, lang="ja"):
        type(self).inference_count += 1
        return 8000, np.array([-1.5, -0.5, 0.0, 0.5, 1.5], dtype=np.float64)


class FailingEngine(SuccessfulEngine):
    def infer(self, ref_wav, ref_text, target_text, lang="ja"):
        raise RuntimeError("deliberate item failure")
