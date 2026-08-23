import json
import sys
from datetime import datetime, timezone

from config import ConfigurationError, parse_config
from utils.audio_handler import ReferenceHandler
from utils.audio_io import write_validated_wav


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write_manifest(output, manifest):
    output.mkdir(parents=True, exist_ok=True)
    path = output / "results_meta.json"
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def run(config, engine_factory=None):
    if config.preflight_only:
        print("Preflight passed", file=sys.stderr)
        return 0

    references = ReferenceHandler(str(config.reference_root)).scan_references()[:config.max_items]
    manifest = {
        "schema_version": 1,
        "started_at": _now(),
        "completed_at": None,
        "status": "running",
        "items": [],
    }
    _write_manifest(config.output, manifest)
    if not references:
        manifest.update(status="failed", completed_at=_now(), error="No valid WAV and TXT reference pairs found")
        _write_manifest(config.output, manifest)
        print("No valid WAV and TXT reference pairs found", file=sys.stderr)
        return 1

    if engine_factory is None:
        from engines.gsv_engine import GSVEngine

        engine_factory = GSVEngine

    engine = engine_factory(
        root=config.gpt_sovits_root,
        weights=config.weights,
        device=config.device,
        precision=config.precision,
        seed=config.seed,
    )
    report_rows = [dict(reference) for reference in references]
    failed = False

    for version in config.versions:
        try:
            engine.load_version(version)
        except Exception as exc:
            failed = True
            for reference in references:
                manifest["items"].append({
                    "name": reference["name"],
                    "version": version,
                    "status": "failed",
                    "error": f"Model load failed: {exc}",
                })
            print(f"Model {version} failed to load: {exc}", file=sys.stderr)
            continue

        for row, reference in zip(report_rows, references):
            destination = config.output / "generated" / version / f"{reference['name']}_{version}.wav"
            item = {"name": reference["name"], "version": version, "status": "running"}
            manifest["items"].append(item)
            try:
                sample_rate, audio = engine.infer(
                    reference["wav_path"], reference["text"], config.target_text, lang=config.language
                )
                if audio is None:
                    raise RuntimeError("Inference returned no audio")
                item["audio"] = write_validated_wav(
                    destination, audio, sample_rate, policy=config.audio_policy
                )
                item["status"] = "complete"
                row[version] = str(destination)
            except Exception as exc:
                failed = True
                item.update(status="failed", error=str(exc))
                print(f"{reference['name']} ({version}) failed: {exc}", file=sys.stderr)
            finally:
                _write_manifest(config.output, manifest)

    manifest["completed_at"] = _now()
    complete_count = sum(item["status"] == "complete" for item in manifest["items"])
    expected_count = len(references) * len(config.versions)
    manifest["status"] = "complete" if not failed and complete_count == expected_count else "failed"
    _write_manifest(config.output, manifest)

    from utils.report_gen import ReportGenerator

    report_path = ReportGenerator(str(config.output)).generate_html(report_rows, config.versions)
    print(f"Report: {report_path}")
    return 0 if manifest["status"] == "complete" else 1


def main(argv=None):
    try:
        return run(parse_config(argv))
    except ConfigurationError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Evaluation interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
