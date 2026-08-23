import hashlib
import json
import runpy
import sys
import types
import wave
from dataclasses import replace

import numpy as np
import pytest

import batch_runner
import config as config_module
from config import ConfigurationError, parse_config, validate_config
from engines.gsv_engine import GSVEngine
from tests.helpers import FailingEngine, SuccessfulEngine, make_config, write_wave
from utils.artifacts import probe_wav, sha256_file, validate_resume
from utils.audio_handler import ReferenceHandler
from utils.audio_io import AudioValidationError, prepare_pcm16, write_validated_wav


def test_parse_config_reads_prompt_and_weight_files(tmp_path, monkeypatch):
    base = make_config(tmp_path)
    prompt = tmp_path / "target.txt"
    prompt.write_text(" file prompt \n", encoding="utf-8")
    weights = tmp_path / "weights.json"
    weights.write_text(json.dumps(base.weights), encoding="utf-8")
    monkeypatch.setattr(config_module, "validate_config", lambda value: None)

    parsed = parse_config([
        "--reference-root", str(base.reference_root), "--output", str(base.output),
        "--gpt-sovits-root", str(base.gpt_sovits_root), "--target-file", str(prompt),
        "--weights-file", str(weights), "--version", "v2", "--language", " ja ",
        "--seed", "9", "--max-items", "2", "--max-target-chars", "50",
        "--audio-policy", "reject",
        "--existing-output", "overwrite", "--reference-policy", "omit", "--preflight",
    ])

    assert parsed.target_text == "file prompt"
    assert parsed.language == "ja"
    assert parsed.versions == ("v2",)
    assert parsed.seed == 9 and parsed.max_items == 2 and parsed.max_target_chars == 50
    assert parsed.preflight_only is True


def test_parse_config_supports_text_defaults_and_reports_file_errors(tmp_path, monkeypatch):
    base = make_config(tmp_path)
    monkeypatch.setattr(config_module, "validate_config", lambda value: None)
    parsed = parse_config([
        "--reference-root", str(base.reference_root), "--output", str(base.output),
        "--gpt-sovits-root", str(base.gpt_sovits_root), "--target-text", " prompt ",
    ])
    assert parsed.target_text == "prompt"
    assert parsed.versions == config_module.SUPPORTED_VERSIONS

    missing = tmp_path / "missing.txt"
    with pytest.raises(ConfigurationError, match="Cannot read target prompt"):
        parse_config([
            "--reference-root", ".", "--output", ".", "--gpt-sovits-root", ".",
            "--target-file", str(missing),
        ])
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Cannot read weights file"):
        config_module._load_json(broken, "weights file")


def test_validate_config_aggregates_input_mapping_and_cuda_errors(tmp_path, monkeypatch):
    base = make_config(tmp_path)
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(config_module.importlib.util, "find_spec", lambda name: object())
    invalid = replace(
        base,
        reference_root=tmp_path / "absent-references",
        gpt_sovits_root=tmp_path / "absent-model",
        target_text="",
        language="",
        max_items=0,
        max_target_chars=0,
        device="cuda",
        versions=("missing", "partial"),
        weights={"partial": {"gpt": ""}},
    )

    with pytest.raises(ConfigurationError) as raised:
        validate_config(invalid)

    message = str(raised.value)
    for expected in (
        "Reference root is not a directory", "GPT-SoVITS root is not a directory",
        "Target prompt must not be empty", "Language must not be empty",
        "--max-items must be at least 1", "CUDA was requested",
        "--max-target-chars must be at least 1",
        "No weight mapping for model version missing", "Missing gpt weight",
        "Missing sovits weight",
    ):
        assert expected in message


def test_validate_config_accepts_absolute_weight_paths(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    absolute = {
        "v2": {
            "gpt": str(config.gpt_sovits_root / "gpt.ckpt"),
            "sovits": str(config.gpt_sovits_root / "sovits.pth"),
        }
    }
    monkeypatch.setattr(config_module.importlib.util, "find_spec", lambda name: object())
    validate_config(replace(config, weights=absolute))


def install_engine_contract(monkeypatch, cuda_available=True):
    events = []

    class Cuda:
        @staticmethod
        def is_available():
            return cuda_available

        @staticmethod
        def manual_seed_all(seed):
            events.append(("cuda-seed", seed))

        @staticmethod
        def empty_cache():
            events.append(("empty-cache",))

    torch = types.ModuleType("torch")
    torch.cuda = Cuda
    torch.manual_seed = lambda seed: events.append(("seed", seed))

    class TTSConfig:
        def __init__(self, values):
            self.values = values

    class TTS:
        responses = [(22050, [0.0])]

        def __init__(self, configuration):
            self.configuration = configuration

        def run(self, inputs):
            events.append(("inputs", inputs))
            return iter(type(self).responses)

    package = types.ModuleType("GPT_SoVITS")
    infer_package = types.ModuleType("GPT_SoVITS.TTS_infer_pack")
    tts_module = types.ModuleType("GPT_SoVITS.TTS_infer_pack.TTS")
    tts_module.TTS = TTS
    tts_module.TTS_Config = TTSConfig
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "GPT_SoVITS", package)
    monkeypatch.setitem(sys.modules, "GPT_SoVITS.TTS_infer_pack", infer_package)
    monkeypatch.setitem(sys.modules, "GPT_SoVITS.TTS_infer_pack.TTS", tts_module)
    return events, TTS


def test_engine_configures_switches_releases_and_infers(tmp_path, monkeypatch):
    events, tts_class = install_engine_contract(monkeypatch)
    absolute = tmp_path / "absolute.pth"
    weights = {
        "v2": {"gpt": "gpt.ckpt", "sovits": "sovits.pth"},
        "v3": {"gpt": str(absolute), "sovits": "v3.pth"},
    }
    engine = GSVEngine(tmp_path, weights, device="cuda", precision="float16", seed=12)
    engine.load_version("v2")
    first = engine.tts_pipeline
    engine.load_version("v2")
    engine.load_version("v3")

    rate, samples = engine.infer("ref.wav", "reference", "target", lang="ja")
    assert (rate, samples) == (22050, [0.0])
    assert first is not engine.tts_pipeline
    assert engine.tts_pipeline.configuration.values["t2s_weights_path"] == str(absolute)
    assert ("seed", 12) in events and ("cuda-seed", 12) in events and ("empty-cache",) in events
    assert next(event for event in events if event[0] == "inputs")[1]["sample_steps"] == 32

    tts_class.responses = []
    assert engine.infer("r", "p", "t") == (None, None)
    with pytest.raises(ValueError, match="Unsupported version"):
        engine.load_version("unknown")


def test_batch_preflight_empty_inputs_and_default_engine_factory(tmp_path, monkeypatch, capsys):
    config = make_config(tmp_path, preflight_only=True)
    assert batch_runner.run(config) == 0
    empty = replace(config, preflight_only=False)
    empty.reference_root.joinpath("voice.txt").unlink()
    assert batch_runner.run(empty, engine_factory=SuccessfulEngine) == 1
    assert "No valid WAV" in capsys.readouterr().err

    complete = make_config(tmp_path / "default")
    monkeypatch.setattr("engines.gsv_engine.GSVEngine", SuccessfulEngine)
    assert batch_runner.run(complete) == 0


@pytest.mark.parametrize("engine_factory,error", [
    (lambda **kwargs: (_ for _ in ()).throw(RuntimeError("init")), "Engine initialization failed"),
    (type("LoadFailure", (SuccessfulEngine,), {"load_version": lambda self, version: (_ for _ in ()).throw(RuntimeError("load"))}), "Model load failed"),
])
def test_batch_records_engine_and_model_failures(tmp_path, engine_factory, error):
    config = make_config(tmp_path)
    assert batch_runner.run(config, engine_factory=engine_factory) == 1
    manifest = json.loads((config.output / "results_meta.json").read_text(encoding="utf-8"))
    assert error in manifest["items"][0]["error"]


def test_batch_overwrite_interrupt_none_audio_and_report_failure(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    assert batch_runner.run(config, engine_factory=SuccessfulEngine) == 0
    assert batch_runner.run(replace(config, existing_output="overwrite"), engine_factory=SuccessfulEngine) == 0

    class NoneEngine(SuccessfulEngine):
        def infer(self, *args, **kwargs):
            return 8000, None

    none_config = make_config(tmp_path / "none")
    assert batch_runner.run(none_config, engine_factory=NoneEngine) == 1
    assert "Inference returned no audio" in json.loads(
        (none_config.output / "results_meta.json").read_text(encoding="utf-8")
    )["items"][0]["error"]

    class InterruptedEngine(SuccessfulEngine):
        def infer(self, *args, **kwargs):
            raise KeyboardInterrupt()

    interrupted = make_config(tmp_path / "interrupted")
    with pytest.raises(KeyboardInterrupt):
        batch_runner.run(interrupted, engine_factory=InterruptedEngine)
    assert json.loads((interrupted.output / "results_meta.json").read_text())["status"] == "interrupted"

    report_failure = make_config(tmp_path / "report")
    monkeypatch.setattr("utils.report_gen.ReportGenerator.generate_html", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("report")))
    assert batch_runner.run(report_failure, engine_factory=SuccessfulEngine) == 1


def test_batch_rejects_unpermitted_existing_artifact(tmp_path):
    config = make_config(tmp_path, existing_output="invalid")
    destination = config.output / "generated" / "v2" / "voice_v2.wav"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"existing")
    assert batch_runner.run(config, engine_factory=SuccessfulEngine) == 1


def test_main_reports_configuration_error_and_script_guard(monkeypatch, capsys):
    monkeypatch.setattr(batch_runner, "parse_config", lambda argv: (_ for _ in ()).throw(ConfigurationError("bad config")))
    assert batch_runner.main([]) == 2
    assert "bad config" in capsys.readouterr().err

    monkeypatch.setattr(sys, "argv", ["batch_runner.py", "--help"])
    with pytest.raises(SystemExit) as exited:
        runpy.run_module("batch_runner", run_name="__main__")
    assert exited.value.code == 0


def test_artifact_probe_resume_and_audio_failure_branches(tmp_path, monkeypatch):
    empty = tmp_path / "empty.wav"
    write_wave(empty, samples=[])
    with pytest.raises(ValueError, match="invalid stream metadata"):
        probe_wav(empty)
    eight_bit = tmp_path / "eight.wav"
    with wave.open(str(eight_bit), "wb") as output:
        output.setparams((1, 1, 8000, 1, "NONE", "not compressed"))
        output.writeframes(b"\x00")
    with pytest.raises(ValueError, match="16-bit"):
        probe_wav(eight_bit)

    missing = tmp_path / "missing.wav"
    assert validate_resume(missing, {"status": "complete", "fingerprint": "same"}, "same")[1] == "artifact is missing"
    corrupt = tmp_path / "corrupt.wav"
    corrupt.write_bytes(b"not-wave")
    previous = {
        "status": "complete", "fingerprint": "same",
        "sha256": hashlib.sha256(b"not-wave").hexdigest(),
    }
    assert validate_resume(corrupt, previous, "same")[1].startswith("artifact decode failed")
    assert sha256_file(corrupt) == previous["sha256"]

    with pytest.raises(AudioValidationError, match="Invalid sample rate"):
        prepare_pcm16([0], 0)
    with pytest.raises(AudioValidationError, match="one- or two-dimensional"):
        prepare_pcm16(np.zeros((1, 1, 1)), 8000)
    monkeypatch.setattr("soundfile.info", lambda path: types.SimpleNamespace(
        samplerate=1, channels=1, frames=1, subtype="PCM_16"
    ))
    with pytest.raises(AudioValidationError, match="Decoder validation failed"):
        write_validated_wav(tmp_path / "mismatch.wav", [0, 0], 8000)


def test_reference_handler_real_missing_and_malformed_inputs(tmp_path, capsys):
    assert ReferenceHandler(str(tmp_path / "missing")).scan_references() == []
    malformed = tmp_path / "bad.wav"
    malformed.write_bytes(b"bad")
    (tmp_path / "bad.txt").write_bytes(b"\xff")
    references = ReferenceHandler(str(tmp_path)).scan_references()
    assert references[0]["text"] == ""
    assert references[0]["duration"] == 0
    assert references[0]["status"] == "Length Issue"
    assert "Error reading" in capsys.readouterr().err
