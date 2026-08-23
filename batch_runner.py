import json
import sys
from datetime import datetime, timezone

from config import ConfigurationError, parse_config
from utils.artifacts import (
    completed_artifact,
    quarantine,
    redact,
    run_fingerprint,
    validate_resume,
)
from utils.audio_handler import ReferenceHandler
from utils.audio_io import write_validated_wav


def _now():
    return datetime.now(timezone.utc).isoformat()


def _manifest_path(output):
    return output / "results_meta.json"


def _read_manifest(output):
    try:
        value = json.loads(_manifest_path(output).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_manifest(output, manifest):
    output.mkdir(parents=True, exist_ok=True)
    path = _manifest_path(output)
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _new_manifest(config):
    return {
        "schema_version": 2,
        "started_at": _now(),
        "completed_at": None,
        "status": "running",
        "configuration": {
            "versions": list(config.versions),
            "language": config.language,
            "device": config.device,
            "precision": config.precision,
            "seed": config.seed,
            "max_items": config.max_items,
            "max_target_chars": config.max_target_chars,
            "audio_policy": config.audio_policy,
            "reference_policy": config.reference_policy,
        },
        "items": [],
    }


def _previous_items(manifest):
    return {
        (item.get("name"), item.get("version")): item
        for item in manifest.get("items", [])
        if isinstance(item, dict)
    }


def _failed_item(reference, version, error, config, **details):
    item = {
        "name": reference["name"],
        "version": version,
        "status": "failed",
        "error": redact(error, config),
    }
    item.update(details)
    return item


def run(config, engine_factory=None):
    if config.preflight_only:
        print("Preflight passed", file=sys.stderr)
        return 0

    previous_manifest = _read_manifest(config.output)
    if previous_manifest and config.existing_output == "fail":
        print("Output already has a run manifest; choose --existing-output resume or overwrite", file=sys.stderr)
        return 1
    previous = _previous_items(previous_manifest) if config.existing_output == "resume" else {}
    references = ReferenceHandler(str(config.reference_root)).scan_references()[:config.max_items]
    manifest = _new_manifest(config)
    _write_manifest(config.output, manifest)
    if not references:
        manifest.update(status="failed", completed_at=_now(), error="No valid WAV and TXT reference pairs found")
        _write_manifest(config.output, manifest)
        print(manifest["error"], file=sys.stderr)
        return 1

    if engine_factory is None:
        from engines.gsv_engine import GSVEngine

        engine_factory = GSVEngine

    report_rows = [dict(reference) for reference in references]
    failed = False
    engine = None

    try:
        for version in config.versions:
            pending = []
            for row, reference in zip(report_rows, references):
                destination = config.output / "generated" / version / f"{reference['name']}_{version}.wav"
                fingerprint = run_fingerprint(config, reference, version)
                old_item = previous.get((reference["name"], version))

                if config.existing_output == "resume":
                    reusable, reason = validate_resume(destination, old_item, fingerprint)
                    if reusable:
                        item = dict(old_item)
                        item["resumed"] = True
                        manifest["items"].append(item)
                        row[version] = str(destination)
                        continue
                    quarantined = quarantine(destination, config.output)
                    pending.append((row, reference, destination, fingerprint, reason, quarantined))
                elif destination.exists() and config.existing_output == "overwrite":
                    destination.unlink()
                    pending.append((row, reference, destination, fingerprint, None, None))
                elif destination.exists():
                    error = "artifact exists but no output policy permits replacement"
                    manifest["items"].append(_failed_item(reference, version, error, config))
                    print(f"{reference['name']} ({version}) failed: {error}", file=sys.stderr)
                    failed = True
                else:
                    pending.append((row, reference, destination, fingerprint, None, None))

            if not pending:
                continue
            if engine is None:
                try:
                    engine = engine_factory(
                        root=config.gpt_sovits_root,
                        weights=config.weights,
                        device=config.device,
                        precision=config.precision,
                        seed=config.seed,
                    )
                except Exception as exc:
                    for _, reference, _, _, resume_error, quarantined in pending:
                        manifest["items"].append(_failed_item(
                            reference, version, f"Engine initialization failed: {exc}", config,
                            resume_error=resume_error, quarantined=quarantined,
                        ))
                    print(f"Engine initialization failed: {redact(exc, config)}", file=sys.stderr)
                    failed = True
                    continue
            try:
                engine.load_version(version)
            except Exception as exc:
                for _, reference, _, _, resume_error, quarantined in pending:
                    manifest["items"].append(_failed_item(
                        reference, version, f"Model load failed: {exc}", config,
                        resume_error=resume_error, quarantined=quarantined,
                    ))
                print(f"Model {version} failed to load: {redact(exc, config)}", file=sys.stderr)
                failed = True
                continue

            for row, reference, destination, fingerprint, resume_error, quarantined in pending:
                try:
                    sample_rate, audio = engine.infer(
                        reference["wav_path"], reference["text"], config.target_text, lang=config.language
                    )
                    if audio is None:
                        raise RuntimeError("Inference returned no audio")
                    audio_metadata = write_validated_wav(
                        destination, audio, sample_rate, policy=config.audio_policy
                    )
                    relative = destination.relative_to(config.output)
                    item = completed_artifact(destination, relative, fingerprint, audio_metadata)
                    item.update(name=reference["name"], version=version, resumed=False)
                    if resume_error:
                        item["resume_error"] = resume_error
                    if quarantined:
                        item["quarantined"] = quarantined
                    manifest["items"].append(item)
                    row[version] = str(destination)
                except Exception as exc:
                    failed = True
                    item = _failed_item(
                        reference, version, exc, config,
                        fingerprint=fingerprint, resume_error=resume_error, quarantined=quarantined,
                    )
                    manifest["items"].append(item)
                    print(f"{reference['name']} ({version}) failed: {item['error']}", file=sys.stderr)
                finally:
                    _write_manifest(config.output, manifest)
    except KeyboardInterrupt:
        manifest.update(status="interrupted", completed_at=_now())
        _write_manifest(config.output, manifest)
        raise

    manifest["completed_at"] = _now()
    complete_count = sum(item["status"] == "complete" for item in manifest["items"])
    expected_count = len(references) * len(config.versions)
    manifest["status"] = "complete" if not failed and complete_count == expected_count else "failed"
    _write_manifest(config.output, manifest)

    try:
        from utils.report_gen import ReportGenerator

        report_path = ReportGenerator(str(config.output)).generate_html(
            report_rows, config.versions, reference_policy=config.reference_policy
        )
        print(f"Report: {report_path}")
    except Exception as exc:
        manifest.update(status="failed", report_error=redact(exc, config))
        _write_manifest(config.output, manifest)
        print(f"Report generation failed: {manifest['report_error']}", file=sys.stderr)
        return 1
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
