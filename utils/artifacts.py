import hashlib
import json
import shutil
import wave
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe_wav(path):
    with wave.open(str(path), "rb") as audio:
        metadata = {
            "sample_rate": audio.getframerate(),
            "channels": audio.getnchannels(),
            "frames": audio.getnframes(),
            "sample_width": audio.getsampwidth(),
        }
        if metadata["sample_rate"] < 1 or metadata["channels"] < 1 or metadata["frames"] < 1:
            raise ValueError("WAV contains invalid stream metadata")
        if metadata["sample_width"] != 2:
            raise ValueError("WAV is not 16-bit PCM")
        audio.readframes(metadata["frames"])
    return metadata


def _weight_identity(config, version):
    identity = {}
    for kind, value in config.weights[version].items():
        path = Path(value)
        if not path.is_absolute():
            path = config.gpt_sovits_root / path
        stat = path.stat()
        identity[kind] = {"file": path.name, "bytes": stat.st_size, "modified_ns": stat.st_mtime_ns}
    return identity


def run_fingerprint(config, reference, version):
    payload = {
        "version": version,
        "weights": _weight_identity(config, version),
        "reference_sha256": sha256_file(reference["wav_path"]),
        "reference_text": reference["text"],
        "target_text": config.target_text,
        "language": config.language,
        "device": config.device,
        "precision": config.precision,
        "seed": config.seed,
        "max_target_chars": config.max_target_chars,
        "audio_policy": config.audio_policy,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def completed_artifact(path, relative_path, fingerprint, audio_metadata):
    return {
        "status": "complete",
        "path": Path(relative_path).as_posix(),
        "sha256": sha256_file(path),
        "bytes": Path(path).stat().st_size,
        "fingerprint": fingerprint,
        "audio": audio_metadata,
    }


def validate_resume(path, previous, fingerprint):
    if not previous or previous.get("status") != "complete":
        return False, "manifest item is not complete"
    if previous.get("fingerprint") != fingerprint:
        return False, "run fingerprint changed"
    if not Path(path).is_file():
        return False, "artifact is missing"
    try:
        if sha256_file(path) != previous.get("sha256"):
            return False, "artifact checksum changed"
        if Path(path).stat().st_size != previous.get("bytes"):
            return False, "artifact byte count changed"
        decoded = probe_wav(path)
        expected_audio = previous.get("audio", {})
        for key in ("sample_rate", "channels", "frames"):
            if expected_audio.get(key) != decoded.get(key):
                return False, f"artifact {key} metadata changed"
    except (OSError, EOFError, wave.Error, ValueError) as exc:
        return False, f"artifact decode failed: {exc}"
    return True, None


def quarantine(path, output_root):
    source = Path(path)
    if not source.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = Path(output_root) / "quarantine" / f"{source.name}.{stamp}.invalid"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), destination)
    return destination.relative_to(output_root).as_posix()


def redact(value, config):
    text = str(value)
    replacements = {
        str(config.reference_root): "<reference-root>",
        str(config.gpt_sovits_root): "<gpt-sovits-root>",
        str(config.output): "<output>",
        str(Path.home()): "<home>",
    }
    for private, public in sorted(replacements.items(), key=lambda pair: len(pair[0]), reverse=True):
        if private:
            text = text.replace(private, public)
    return text
