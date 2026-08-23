import json
import wave

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
