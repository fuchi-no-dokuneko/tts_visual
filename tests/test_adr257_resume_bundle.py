import json
import re
import shutil
import wave
from dataclasses import replace
from pathlib import Path

import batch_runner
from tests.helpers import SuccessfulEngine, make_config


def load_manifest(config):
    return json.loads((config.output / "results_meta.json").read_text(encoding="utf-8"))


def test_resume_reuses_only_matching_decodable_artifact(tmp_path):
    SuccessfulEngine.inference_count = 0
    config = make_config(tmp_path)
    assert batch_runner.run(config, engine_factory=SuccessfulEngine) == 0
    assert SuccessfulEngine.inference_count == 1

    assert batch_runner.run(config, engine_factory=SuccessfulEngine) == 0

    item = load_manifest(config)["items"][0]
    assert SuccessfulEngine.inference_count == 1
    assert item["status"] == "complete"
    assert item["resumed"] is True


def test_corrupt_resume_file_is_quarantined_and_regenerated(tmp_path):
    SuccessfulEngine.inference_count = 0
    config = make_config(tmp_path)
    assert batch_runner.run(config, engine_factory=SuccessfulEngine) == 0
    old_item = load_manifest(config)["items"][0]
    artifact = config.output / old_item["path"]
    artifact.write_bytes(b"partial")

    assert batch_runner.run(config, engine_factory=SuccessfulEngine) == 0

    item = load_manifest(config)["items"][0]
    assert SuccessfulEngine.inference_count == 2
    assert item["resume_error"] == "artifact checksum changed"
    assert (config.output / item["quarantined"]).read_bytes() == b"partial"
    with wave.open(str(config.output / item["path"]), "rb") as decoded:
        assert decoded.getnframes() == 5


def test_changed_parameter_fingerprint_forces_regeneration(tmp_path):
    SuccessfulEngine.inference_count = 0
    config = make_config(tmp_path)
    assert batch_runner.run(config, engine_factory=SuccessfulEngine) == 0

    changed = replace(config, target_text="a different target")
    assert batch_runner.run(changed, engine_factory=SuccessfulEngine) == 0

    item = load_manifest(changed)["items"][0]
    assert SuccessfulEngine.inference_count == 2
    assert item["resume_error"] == "run fingerprint changed"
    assert item["quarantined"].startswith("quarantine/")


def test_bundle_remains_decodable_after_move_and_has_no_absolute_private_paths(tmp_path):
    config = make_config(tmp_path)
    assert batch_runner.run(config, engine_factory=SuccessfulEngine) == 0
    private_paths = (str(tmp_path), str(Path.home()))

    moved = tmp_path.parent / f"{tmp_path.name}-moved-bundle"
    shutil.move(config.output, moved)
    html = (moved / "index.html").read_text(encoding="utf-8")
    run_json = (moved / "results_meta.json").read_text(encoding="utf-8")
    report_json = (moved / "report_manifest.json").read_text(encoding="utf-8")

    for private in private_paths:
        assert private not in html
        assert private not in run_json
        assert private not in report_json
    sources = re.findall(r'src="([^"]+\.wav)"', html)
    assert len(sources) == 2
    for source in sources:
        assert not Path(source).is_absolute()
        with wave.open(str(moved / source), "rb") as decoded:
            assert decoded.getnframes() > 0


def test_existing_output_fail_policy_refuses_manifest(tmp_path, capsys):
    config = make_config(tmp_path)
    assert batch_runner.run(config, engine_factory=SuccessfulEngine) == 0

    refused = replace(config, existing_output="fail")
    assert batch_runner.run(refused, engine_factory=SuccessfulEngine) == 1
    assert "Output already has a run manifest" in capsys.readouterr().err
