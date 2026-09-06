#!/usr/bin/env python3
import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--junit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    root = ElementTree.parse(args.junit).getroot()
    cases = [
        case for case in root.iter("testcase")
        if "test_adr258_real_process" in case.get("classname", "")
    ]
    passed = bool(cases) and all(
        case.find("failure") is None and case.find("error") is None and case.find("skipped") is None
        for case in cases
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    ).stdout.strip()
    evidence = {
        "schema_version": 1,
        "repository": "fuchi-no-dokuneko/tts_visual",
        "commit": commit,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "tests": [case.get("name") for case in cases],
    }
    args.output.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
