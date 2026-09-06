import json
import wave
from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf

import batch_runner
from config import ConfigurationError, validate_config
from tests.helpers import FailingEngine, SuccessfulEngine, make_config
from utils.audio_io import AudioValidationError, prepare_pcm16, write_validated_wav


def test_preflight_reports_missing_dependencies_models_and_invalid_cpu_precision(tmp_path, monkeypatch):
    config = make_config(tmp_path, precision="float16")
    config.gpt_sovits_root.joinpath("gpt.ckpt").unlink()
    real_find_spec = __import__("importlib.util").util.find_spec
    monkeypatch.setattr(
        "config.importlib.util.find_spec",
        lambda name: None if name == "torch" else real_find_spec(name),
    )

    with pytest.raises(ConfigurationError) as raised:
        validate_config(config)

    message = str(raised.value)
    assert "CPU execution supports only --precision float32" in message
    assert "Missing Python dependency 'torch'" in message
    assert "Missing gpt weight for v2" in message


def test_preflight_validates_prompt_limit_output_and_weight_shape(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    output_file = tmp_path / "not-a-directory"
    output_file.write_text("occupied", encoding="utf-8")
    monkeypatch.setattr("config.importlib.util.find_spec", lambda name: object())
    invalid = replace(
        config,
        output=output_file,
        target_text="too long",
        max_target_chars=3,
        weights=[],
    )

    with pytest.raises(ConfigurationError) as raised:
        validate_config(invalid)

    message = str(raised.value)
    assert "Output path is not a directory" in message
    assert "Target prompt has 8 characters; limit is 3" in message
    assert "Weights configuration must be a JSON object" in message


def test_preflight_requires_reference_pair_model_entry_writable_output_and_path_values(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    config.reference_root.joinpath("voice.txt").unlink()
    config.gpt_sovits_root.joinpath("GPT_SoVITS/TTS_infer_pack/TTS.py").unlink()
    monkeypatch.setattr("config.importlib.util.find_spec", lambda name: object())
    monkeypatch.setattr("config.os.access", lambda path, mode: False)
    invalid = replace(
        config,
        output=tmp_path / "new-output",
        weights={"v2": {"gpt": 123, "sovits": "sovits.pth"}},
    )

    with pytest.raises(ConfigurationError) as raised:
        validate_config(invalid)

    message = str(raised.value)
    assert "no matching WAV and TXT pair" in message
    assert "Python entry point is missing" in message
    assert "Output parent is not writable" in message
    assert "Invalid gpt weight path" in message


def test_float_audio_is_clipped_without_integer_wraparound_and_decodes(tmp_path):
    destination = tmp_path / "clipped.wav"
    source = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])

    metadata = write_validated_wav(destination, source, 16000, policy="clip")
    decoded, sample_rate = sf.read(destination, dtype="int16")
    with wave.open(str(destination), "rb") as header:
        assert header.getsampwidth() == 2
        assert header.getnchannels() == 1
        assert header.getframerate() == 16000

    assert sample_rate == 16000
    assert metadata["clipped_samples"] == 2
    assert decoded.tolist() == [-32767, -32767, -16384, 0, 16384, 32767, 32767]


@pytest.mark.parametrize("samples", [[float("nan")], [float("inf")]])
def test_nonfinite_audio_is_rejected(samples):
    with pytest.raises(AudioValidationError, match="NaN or infinite"):
        prepare_pcm16(samples, 8000)


def test_reject_policy_names_out_of_range_samples():
    with pytest.raises(AudioValidationError, match="2 samples"):
        prepare_pcm16([-1.1, 1.1], 8000, policy="reject")


def test_failed_item_sets_nonzero_result_and_names_item(tmp_path, capsys):
    config = make_config(tmp_path)

    result = batch_runner.run(config, engine_factory=FailingEngine)

    manifest = json.loads((config.output / "results_meta.json").read_text(encoding="utf-8"))
    assert result == 1
    assert manifest["status"] == "failed"
    assert manifest["items"][0]["name"] == "voice"
    assert "voice (v2) failed" in capsys.readouterr().err


def test_complete_run_exits_zero_only_after_decoder_validated_artifact(tmp_path):
    config = make_config(tmp_path)

    result = batch_runner.run(config, engine_factory=SuccessfulEngine)

    manifest = json.loads((config.output / "results_meta.json").read_text(encoding="utf-8"))
    assert result == 0
    assert manifest["status"] == "complete"
    assert manifest["items"][0]["audio"]["pcm_subtype"] == "PCM_16"
    assert (config.output / manifest["items"][0]["path"]).is_file()


def test_main_returns_interrupt_status(monkeypatch, capsys):
    monkeypatch.setattr(batch_runner, "parse_config", lambda argv: object())
    monkeypatch.setattr(batch_runner, "run", lambda config: (_ for _ in ()).throw(KeyboardInterrupt()))

    assert batch_runner.main([]) == 130
    assert "interrupted" in capsys.readouterr().err
