import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
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
    context.acceptance_suite = os.environ.get("ACCEPTANCE_SUITE", "daily")
    context.uat_session = SimpleNamespace(evaluated=False)
    context.run_evidence = {
        "schema_version": 1,
        "repository": "fuchi-no-dokuneko/tts_visual",
        "commit": commit,
        "suite": context.acceptance_suite,
        "started_at": now(),
        "completed_at": None,
        "overall": False,
        "scenarios": [],
        "diagnostics": [],
    }


def before_scenario(context, scenario):
    context.scenario_diagnostics = []
    context.scenario_steps = []


def after_step(context, step):
    status = getattr(step.status, "name", str(step.status))
    context.scenario_steps.append({
        "name": step.name,
        "passed": status == "passed",
        "duration_ms": round(float(step.duration or 0) * 1000),
    })
    if step.exception:
        context.scenario_diagnostics.append(f"{step.name}: {step.exception}")


def after_scenario(context, scenario):
    process = getattr(context, "demo_browser_process", None)
    if process and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    context.demo_browser_process = None
    if getattr(context, "demo_recording", False):
        executable = os.environ.get("DEMO_RECORD_STOP_COMMAND")
        if executable:
            stopped = subprocess.run(
                [executable],
                env={
                    **os.environ,
                    "DEMO_SUITE": context.acceptance_suite,
                    "DEMO_REPOSITORY": "tts_visual",
                },
                timeout=60,
                check=False,
            )
            if stopped.returncode:
                context.scenario_diagnostics.append(
                    f"recording cleanup: DEMO_RECORD_STOP_COMMAND exited with {stopped.returncode}"
                )
        context.demo_recording = False
    status = getattr(scenario.status, "name", str(scenario.status))
    passed = status == "passed" and not context.scenario_diagnostics
    diagnostics = list(context.scenario_diagnostics)
    context.run_evidence["scenarios"].append({
        "feature": str(Path(scenario.filename).resolve().relative_to(REPOSITORY)),
        "name": scenario.name,
        "passed": passed,
        "duration_ms": round(float(scenario.duration or 0) * 1000),
        "steps": list(context.scenario_steps),
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
    for feature in dict.fromkeys(item["feature"] for item in scenarios):
        file_node = ElementTree.SubElement(root, "file", path=feature)
        for item in (record for record in scenarios if record["feature"] == feature):
            case = ElementTree.SubElement(
                file_node, "testCase", name=item["name"], duration=str(item["duration_ms"])
            )
            if not item["passed"]:
                failure = ElementTree.SubElement(case, "failure", message="Acceptance scenario failed")
                failure.text = "\n".join(item["diagnostics"]) or "Scenario did not pass"
    ElementTree.ElementTree(root).write(
        context.artifact_dir / "sonar-test-execution.xml", encoding="utf-8", xml_declaration=True
    )
