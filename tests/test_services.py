"""Service-boundary tests for multimodal descriptions and complete transcripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from slidegeist import services
from slidegeist.transcribe import transcribe_video


def test_llama_completion_sends_image_and_configured_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "slide.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nvisual-oracle")
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
    content = payload["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "Describe"}
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


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
