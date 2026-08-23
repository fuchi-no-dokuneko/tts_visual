from pathlib import Path


class AudioValidationError(ValueError):
    pass


def prepare_pcm16(audio, sample_rate, policy="clip"):
    import numpy as np

    samples = np.asarray(audio, dtype=np.float64)
    if not isinstance(sample_rate, int) or sample_rate < 1:
        raise AudioValidationError(f"Invalid sample rate: {sample_rate}")
    if samples.ndim not in (1, 2) or samples.size == 0:
        raise AudioValidationError("Audio must contain one- or two-dimensional samples")
    if not np.isfinite(samples).all():
        raise AudioValidationError("Audio contains NaN or infinite samples")

    outside = int(np.count_nonzero((samples < -1.0) | (samples > 1.0)))
    if outside and policy == "reject":
        raise AudioValidationError(f"Audio has {outside} samples outside [-1.0, 1.0]")
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = np.rint(clipped * 32767.0).astype("<i2")
    channels = 1 if pcm.ndim == 1 else int(pcm.shape[1])
    frames = int(pcm.shape[0])
    return pcm, {
        "sample_rate": sample_rate,
        "channels": channels,
        "frames": frames,
        "clipped_samples": outside,
        "normalization": "clip-to-unit-range",
        "pcm_subtype": "PCM_16",
    }


def write_validated_wav(path, audio, sample_rate, policy="clip"):
    import soundfile as sf

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pcm, metadata = prepare_pcm16(audio, sample_rate, policy)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    sf.write(temporary, pcm, sample_rate, format="WAV", subtype="PCM_16")
    info = sf.info(temporary)
    if (
        info.samplerate != sample_rate
        or info.channels != metadata["channels"]
        or info.frames != metadata["frames"]
        or info.subtype != "PCM_16"
    ):
        temporary.unlink(missing_ok=True)
        raise AudioValidationError(f"Decoder validation failed for {destination.name}")
    temporary.replace(destination)
    return metadata
