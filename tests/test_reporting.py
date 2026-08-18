import html
import wave

from utils.audio_handler import ReferenceHandler
from utils.report_gen import ReportGenerator


def write_silent_wave(path, seconds=3, sample_rate=8000):
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(b"\x00\x00" * sample_rate * seconds)


def test_reference_handler_matches_wave_and_text(tmp_path):
    wave_path = tmp_path / "voice.wav"
    write_silent_wave(wave_path)
    (tmp_path / "voice.txt").write_text("hello", encoding="utf-8")

    references = ReferenceHandler(str(tmp_path)).scan_references()

    assert len(references) == 1
    assert references[0]["name"] == "voice"
    assert references[0]["duration"] == 3.0
    assert references[0]["status"] == "OK"


def test_report_generator_escapes_user_content(tmp_path):
    output_dir = tmp_path / "report"
    output_dir.mkdir()
    wave_path = tmp_path / "voice.wav"
    generated_path = tmp_path / "generated.wav"
    write_silent_wave(wave_path)
    write_silent_wave(generated_path)
    dangerous = '<script>alert("x")</script>'
    results = [{
        "name": dangerous,
        "text": dangerous,
        "wav_path": str(wave_path),
        "v2": str(generated_path),
    }]

    report_path = ReportGenerator(str(output_dir)).generate_html(results, ["v2"])
    report = open(report_path, encoding="utf-8").read()

    assert dangerous not in report
    assert html.escape(dangerous) in report
