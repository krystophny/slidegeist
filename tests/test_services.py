"""Service-boundary tests for multimodal descriptions and complete transcripts."""

from __future__ import annotations

import base64
import io
import wave
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from slidegeist import services
from slidegeist.transcribe import _chunk_start_offsets, _normalize_transcript, transcribe_video


def test_llama_completion_sends_image_and_configured_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "slide.png"
    Image.new("RGB", (32, 16), color=(17, 43, 91)).save(image)
    captured: dict[str, Any] = {}

    def fake_http_json(url: str, **kwargs: Any) -> dict[str, Any]:
        captured["url"] = url
        captured["payload"] = kwargs["payload"]
        return {"choices": [{"message": {"content": "Visible slide"}}]}

    monkeypatch.setenv("SLIDEGEIST_LLAMACPP_URL", "http://vision.example:8080")
    monkeypatch.setenv("SLIDEGEIST_LLAMACPP_MODEL", "qwen27b")
    monkeypatch.setattr(services, "_http_json", fake_http_json)

    assert services.llama_cpp_complete("Describe", image_path=image) == "Visible slide"
    assert captured["url"] == "http://vision.example:8080/v1/chat/completions"
    payload = captured["payload"]
    assert payload["model"] == "qwen27b"
    assert payload["repeat_penalty"] == 1.1
    content = payload["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "Describe"}
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_llama_completion_limits_only_service_image_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "wide-slide.png"
    Image.new("RGB", (2400, 1200), color=(17, 43, 91)).save(image)
    original = image.read_bytes()
    captured: dict[str, Any] = {}

    def fake_http_json(_: str, **kwargs: Any) -> dict[str, Any]:
        captured["payload"] = kwargs["payload"]
        return {"choices": [{"message": {"content": "Visible slide"}}]}

    monkeypatch.setenv("SLIDEGEIST_LLAMACPP_MODEL", "qwen27b")
    monkeypatch.setenv("SLIDEGEIST_LLAMACPP_MAX_IMAGE_DIMENSION", "800")
    monkeypatch.setattr(services, "_http_json", fake_http_json)

    services.llama_cpp_complete("Describe", image_path=image)

    data_url = captured["payload"]["messages"][0]["content"][1]["image_url"]["url"]
    encoded = data_url.partition(",")[2]
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as sent:
        assert sent.size == (800, 400)
    assert image.read_bytes() == original


def test_transcription_refuses_partial_chunk_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "lecture.mp4"
    video.write_bytes(b"video")
    chunk_a = tmp_path / "a.wav"
    chunk_b = tmp_path / "b.wav"
    chunk_a.write_bytes(b"a")
    chunk_b.write_bytes(b"b")

    monkeypatch.setattr("slidegeist.transcribe.get_video_duration", lambda _: 240.0)
    monkeypatch.setattr("slidegeist.transcribe.extract_audio", lambda *_: None)
    monkeypatch.setattr(
        "slidegeist.transcribe._split_audio_chunks",
        lambda *_: [chunk_a, chunk_b],
    )
    calls = 0

    def fake_transcribe(*_: Any, **__: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ConnectionError("service interrupted")
        return {
            "language": "en",
            "segments": [{"start": 0.0, "end": 1.0, "text": "first", "words": []}],
        }

    monkeypatch.setattr("slidegeist.transcribe.whisper_transcribe", fake_transcribe)

    with pytest.raises(RuntimeError, match="refusing to publish a partial transcript"):
        transcribe_video(video)


def test_transcript_chunk_offsets_follow_pcm_sample_counts(tmp_path: Path) -> None:
    chunks = [tmp_path / "first.wav", tmp_path / "second.wav"]
    for path, frames in zip(chunks, (20_000, 8_000), strict=True):
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16_000)
            stream.writeframes(b"\0\0" * frames)

    assert _chunk_start_offsets(chunks) == [0.0, 1.25]


def test_incompatible_whisper_vad_word_clock_is_rebased() -> None:
    """Words on the VAD-compressed clock are mapped back onto the segment clock."""
    payload = {
        "language": "en",
        "segments": [
            {
                "start": 14.66,
                "end": 18.02,
                "text": "Denmark they did this annual bird",
                "words": [
                    {"word": "Denmark", "start": 12.6, "end": 13.2},
                    {"word": "bird", "start": 14.96, "end": 15.45},
                ],
            },
            {
                "start": 18.02,
                "end": 23.47,
                "text": "deaths by wind turbines",
                "words": [
                    {"word": "deaths", "start": 15.46, "end": 16.1},
                    {"word": "turbines", "start": 18.8, "end": 19.95},
                ],
            },
        ],
    }

    result = _normalize_transcript(payload)

    assert [(segment["start"], segment["end"]) for segment in result["segments"]] == [
        (14.66, 18.02),
        (18.02, 23.47),
    ]

    # Word timing survives, rebased onto each segment's own span instead of
    # being discarded wholesale: word-level speaker assignment depends on it.
    for segment in result["segments"]:
        assert segment["words"], "word timing must be preserved, not dropped"
        previous_end = segment["start"] - 1e-9
        for word in segment["words"]:
            assert segment["start"] - 1e-9 <= word["start"] <= segment["end"] + 1e-9
            assert segment["start"] - 1e-9 <= word["end"] <= segment["end"] + 1e-9
            assert word["start"] >= previous_end - 1e-9
            assert word["end"] >= word["start"]
            previous_end = word["end"]

    # The first and last word of a segment anchor to the segment boundaries.
    first = result["segments"][0]
    assert first["words"][0]["start"] == pytest.approx(14.66)
    assert first["words"][-1]["end"] == pytest.approx(18.02)


def test_unusable_word_timing_is_dropped_per_segment() -> None:
    """Non-monotonic words lose timing for their own segment only."""
    payload = {
        "language": "en",
        "segments": [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "good segment",
                "words": [
                    {"word": "good", "start": 0.0, "end": 1.0},
                    {"word": "segment", "start": 1.0, "end": 2.0},
                ],
            },
            {
                "start": 2.0,
                "end": 4.0,
                "text": "bad segment",
                "words": [
                    {"word": "bad", "start": 3.5, "end": 3.9},
                    {"word": "segment", "start": 2.1, "end": 2.4},
                ],
            },
        ],
    }

    result = _normalize_transcript(payload)

    assert result["segments"][0]["words"], "healthy segment keeps its words"
    assert not result["segments"][1]["words"], "only the broken segment loses timing"
