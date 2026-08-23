import argparse
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_VERSIONS = ("v2", "v2ProPlus", "v3", "v4")
DEFAULT_WEIGHTS = {
    "v2": {
        "gpt": "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
        "sovits": "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
    },
    "v2ProPlus": {
        "gpt": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
        "sovits": "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth",
    },
    "v3": {
        "gpt": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
        "sovits": "GPT_SoVITS/pretrained_models/s2Gv3.pth",
    },
    "v4": {
        "gpt": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
        "sovits": "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth",
    },
}


class ConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class EvaluationConfig:
    reference_root: Path
    output: Path
    gpt_sovits_root: Path
    versions: tuple[str, ...]
    weights: dict
    target_text: str
    language: str
    device: str
    precision: str
    seed: int
    max_items: int
    audio_policy: str
    preflight_only: bool


def build_parser():
    parser = argparse.ArgumentParser(description="Portable GPT-SoVITS model evaluator")
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpt-sovits-root", required=True, type=Path)
    parser.add_argument("--version", action="append", choices=SUPPORTED_VERSIONS, dest="versions")
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--target-text")
    prompt.add_argument("--target-file", type=Path)
    parser.add_argument("--weights-file", type=Path)
    parser.add_argument("--language", default="ja")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--precision", choices=("float16", "float32"), default="float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=100)
    parser.add_argument("--audio-policy", choices=("clip", "reject"), default="clip")
    parser.add_argument("--preflight", action="store_true", help="validate inputs without loading a model")
    return parser


def _load_json(path, label):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Cannot read {label} {path}: {exc}") from exc


def parse_config(argv=None):
    args = build_parser().parse_args(argv)
    if args.target_file:
        try:
            target_text = args.target_file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ConfigurationError(f"Cannot read target prompt {args.target_file}: {exc}") from exc
    else:
        target_text = args.target_text.strip()

    weights = DEFAULT_WEIGHTS
    if args.weights_file:
        weights = _load_json(args.weights_file, "weights file")

    config = EvaluationConfig(
        reference_root=args.reference_root.expanduser().resolve(),
        output=args.output.expanduser().resolve(),
        gpt_sovits_root=args.gpt_sovits_root.expanduser().resolve(),
        versions=tuple(args.versions or SUPPORTED_VERSIONS),
        weights=weights,
        target_text=target_text,
        language=args.language.strip(),
        device=args.device,
        precision=args.precision,
        seed=args.seed,
        max_items=args.max_items,
        audio_policy=args.audio_policy,
        preflight_only=args.preflight,
    )
    validate_config(config)
    return config


def validate_config(config):
    errors = []
    if not config.reference_root.is_dir():
        errors.append(f"Reference root is not a directory: {config.reference_root}")
    if not config.gpt_sovits_root.is_dir():
        errors.append(f"GPT-SoVITS root is not a directory: {config.gpt_sovits_root}")
    if not config.target_text:
        errors.append("Target prompt must not be empty")
    if not config.language:
        errors.append("Language must not be empty")
    if config.max_items < 1:
        errors.append("--max-items must be at least 1")
    if config.device == "cpu" and config.precision != "float32":
        errors.append("CPU execution supports only --precision float32")

    for module in ("numpy", "soundfile", "torch"):
        if importlib.util.find_spec(module) is None:
            errors.append(f"Missing Python dependency '{module}'; install the locked project dependencies")

    if config.device == "cuda" and importlib.util.find_spec("torch") is not None:
        import torch

        if not torch.cuda.is_available():
            errors.append("CUDA was requested but torch.cuda.is_available() is false")

    for version in config.versions:
        version_weights = config.weights.get(version)
        if not isinstance(version_weights, dict):
            errors.append(f"No weight mapping for model version {version}")
            continue
        for kind in ("gpt", "sovits"):
            value = version_weights.get(kind)
            if not value:
                errors.append(f"Missing {kind} weight for model version {version}")
                continue
            path = Path(value)
            if not path.is_absolute():
                path = config.gpt_sovits_root / path
            if not path.is_file():
                errors.append(f"Missing {kind} weight for {version}: {path}")

    if errors:
        raise ConfigurationError("Preflight failed:\n- " + "\n- ".join(errors))
