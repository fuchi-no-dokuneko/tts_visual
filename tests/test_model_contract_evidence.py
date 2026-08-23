import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "model-contract-evidence.py"


def test_model_contract_evidence_requires_and_records_passing_process_tests(tmp_path):
    junit = tmp_path / "junit.xml"
    junit.write_text(
        """<?xml version="1.0"?><testsuites><testsuite>
<testcase classname="tests.test_adr258_real_process" name="test_real_process"/>
</testsuite></testsuites>""",
        encoding="utf-8",
    )
    output = tmp_path / "evidence.json"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--junit", str(junit), "--output", str(output)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert evidence["passed"] is True
    assert evidence["tests"] == ["test_real_process"]


def test_model_contract_evidence_blocks_missing_or_failed_tests(tmp_path):
    for body in (
        "<testsuites><testsuite/></testsuites>",
        "<testsuites><testsuite><testcase classname=\"tests.test_adr258_real_process\" name=\"bad\"><failure/></testcase></testsuite></testsuites>",
    ):
        junit = tmp_path / "junit.xml"
        output = tmp_path / "evidence.json"
        junit.write_text(body, encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "--junit", str(junit), "--output", str(output)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        assert completed.returncode == 1
        assert json.loads(output.read_text(encoding="utf-8"))["passed"] is False
