import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree


REPOSITORY = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def before_all(context):
    artifact_dir = Path(os.environ.get("UAT_ARTIFACT_DIR", REPOSITORY / "uat-artifacts")).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    ).stdout.strip()
    context.artifact_dir = artifact_dir
    context.run_evidence = {
        "schema_version": 1,
        "repository": "fuchi-no-dokuneko/tts_visual",
        "commit": commit,
        "started_at": now(),
        "completed_at": None,
        "overall": False,
        "scenarios": [],
        "diagnostics": [],
    }


def before_scenario(context, scenario):
    context.scenario_diagnostics = []


def after_step(context, step):
    if step.exception:
        context.scenario_diagnostics.append(f"{step.name}: {step.exception}")


def after_scenario(context, scenario):
    status = getattr(scenario.status, "name", str(scenario.status))
    passed = status == "passed"
    diagnostics = list(context.scenario_diagnostics)
    context.run_evidence["scenarios"].append({
        "name": scenario.name,
        "passed": passed,
        "duration_ms": round(float(scenario.duration or 0) * 1000),
        "diagnostics": diagnostics,
    })
    context.run_evidence["diagnostics"].extend(diagnostics)


def after_all(context):
    evidence = context.run_evidence
    scenarios = evidence["scenarios"]
    evidence["completed_at"] = now()
    evidence["overall"] = bool(scenarios) and all(item["passed"] for item in scenarios)
    checklist = context.artifact_dir / "checklist.json"
    checklist.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")

    root = ElementTree.Element("testExecutions", version="1")
    file_node = ElementTree.SubElement(root, "file", path="features/daily_uat.feature")
    for item in scenarios:
        case = ElementTree.SubElement(
            file_node, "testCase", name=item["name"], duration=str(item["duration_ms"])
        )
        if not item["passed"]:
            failure = ElementTree.SubElement(case, "failure", message="Daily UAT failed")
            failure.text = "\n".join(item["diagnostics"]) or "Scenario did not pass"
    ElementTree.ElementTree(root).write(
        context.artifact_dir / "sonar-test-execution.xml", encoding="utf-8", xml_declaration=True
    )
