"""Integration-style tests exercising the high-level pipeline and CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from slidegeist import cli
from slidegeist.pipeline import process_video


def test_process_video_produces_slides_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure process_video writes slides.json and returns paths."""
    # OCR is disabled by default now, no need to mock it

    def fake_detect_scenes(*_: Any, **kwargs: Any) -> list[float]:
        kwargs["diagnostics_path"].write_text(
            json.dumps({"timestamps": [2.0]}), encoding="utf-8"
        )
        return [2.0]

    def fake_extract_slides(
        video_path: Path,
        scene_timestamps: list[float],
        output_dir: Path,
        image_format: str
    ) -> list[tuple[int, float, float, Path]]:
        slides_dir = output_dir / "slides"
        slides_dir.mkdir(parents=True, exist_ok=True)
        paths: list[tuple[int, float, float, Path]] = []
        for index, start in enumerate([0.0, 2.0], start=1):  # 1-based numbering
            end = start + 2.0
            slide_path = slides_dir / f"slide_{index:03d}.{image_format}"
            slide_path.write_bytes(b"fake image")
            paths.append((index, start, end, slide_path))
        return paths

    class FakeTranscriber:
        name = "independent fixture transcriber"
        provider = "fake"
        model = "fake-model"
        provides_speakers = False

        def transcribe(self, *_: Any, **__: Any) -> tuple[dict[str, Any], list[Any]]:
            return (
                {
                    "language": "en",
                    "segments": [
                        {"start": 0.0, "end": 1.0, "text": "Hello", "words": []},
                        {"start": 2.0, "end": 3.0, "text": "World", "words": []},
                    ],
                },
                [],
            )

    class FakeDescriber:
        name = "independent fixture"

        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        def describe(self, *_: Any, **__: Any) -> str:
            return (
                "0. FRAME TYPE\nSLIDE\n\n"
                "1. TITLE\nKnown fixture\n\n"
                "2. TEXT CONTENT\nKnown text\n\n"
                "3. FORMULAS\nNone\n\n"
                "4. VISUAL ELEMENTS\nNone\n\n"
                "5. LAYOUT\nCentered"
            )

    monkeypatch.setattr("slidegeist.pipeline.detect_scenes", fake_detect_scenes)
    monkeypatch.setattr("slidegeist.pipeline.extract_slides", fake_extract_slides)
    monkeypatch.setattr(
        "slidegeist.ai_description.build_ai_describer", FakeDescriber
    )

    video_path = tmp_path / "dummy.mp4"
    video_path.write_bytes(b"fake video content")

    result = process_video(
        video_path=video_path,
        output_dir=tmp_path / "out",
        scene_threshold=0.05,
        min_scene_len=1.0,
        start_offset=0.0,
        model="tiny",
        image_format="png",
        transcriber=FakeTranscriber(),
    )

    slides = result.get("slides")
    assert isinstance(slides, list)
    assert len(slides) == 2

    # Check slides markdown exists (default: slides.md combined file)
    slides_md = result.get("slides_md")
    assert isinstance(slides_md, Path)
    assert slides_md.exists()
    slides_content = slides_md.read_text()
    assert "# Lecture Slides" in slides_content

    # Check combined markdown file contains slides (default mode, no --split)
    output_dir = result.get("output_dir")
    assert isinstance(output_dir, Path)

    # Default mode: single slides.md file with 1-based numbering
    assert "## Slide 1" in slides_content
    assert "Hello" in slides_content
    # OCR disabled by default, no "refined" content expected

    # Should not have separate slide files in default mode
    slide_001_md = output_dir / "slide_001.md"
    assert not slide_001_md.exists()


def test_cli_process_default_invocation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Calling cli.main without subcommand should execute process pipeline."""
    output_dir = tmp_path / "cli-out"

    def fake_process_video(*_: Any, **__: Any) -> dict[str, Any]:
        slides_md = output_dir / "slides.md"
        output_dir.mkdir(parents=True, exist_ok=True)
        slides_md.write_text('# Lecture Slides')
        return {
            "output_dir": output_dir,
            "slides": [],
            "slides_md": slides_md,
        }

    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("slidegeist.cli.process_video", fake_process_video)
    monkeypatch.setattr("slidegeist.cli.check_prerequisites", lambda: None)
    monkeypatch.setattr("slidegeist.cli.resolve_video_path", lambda input_str, output_dir, cookies_from_browser=None: Path(input_str))
    monkeypatch.setattr(sys, "argv", ["slidegeist", str(tmp_path / "input.mp4")])

    cli.main()

    captured = capsys.readouterr()
    assert "Processing complete" in captured.out


def test_cli_defaults_to_local_and_needs_no_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The default run is fully local and must not require any API key.

    Cloud description was reverted as the default: over a full lecture, Gemma
    classified 36-38 of 40 genuine teaching pages as NON-SLIDE, which would make
    the pipeline delete them.
    """
    calls: list[dict[str, Any]] = []

    def fake_process_video(*_: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"output_dir": tmp_path, "slides": [], "slides_md": tmp_path / "slides.md"}

    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.delenv("SLIDEGEIST_MISTRAL_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("slidegeist.cli.process_video", fake_process_video)
    monkeypatch.setattr("slidegeist.cli.check_prerequisites", lambda: None)
    monkeypatch.setattr(
        "slidegeist.cli.resolve_video_path",
        lambda input_str, output_dir, cookies_from_browser=None: Path(input_str),
    )
    monkeypatch.setattr(sys, "argv", ["slidegeist", str(tmp_path / "input.mp4")])

    cli.main()

    assert calls, "a fully local run must proceed without any API key"
    assert calls[0]["provider"] == "whisper"
    assert calls[0]["describer_provider"] == "local"


def test_cli_local_flag_selects_whisper(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--local must not require a Mistral key and must not silently fall back."""
    calls: list[dict[str, Any]] = []

    def fake_process_video(*_: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"output_dir": tmp_path, "slides": [], "slides_md": tmp_path / "slides.md"}

    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.setattr("slidegeist.cli.process_video", fake_process_video)
    monkeypatch.setattr("slidegeist.cli.check_prerequisites", lambda: None)
    monkeypatch.setattr(
        "slidegeist.cli.resolve_video_path",
        lambda input_str, output_dir, cookies_from_browser=None: Path(input_str),
    )
    monkeypatch.setattr(
        sys, "argv", ["slidegeist", "process", str(tmp_path / "input.mp4"), "--local"]
    )

    cli.main()

    assert calls and calls[0]["provider"] == "whisper"
    assert calls[0]["describer_provider"] == "local", (
        "--local must keep slide descriptions on this machine too"
    )


def test_cloud_providers_are_opt_in(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Cloud backends must be reachable, but only when explicitly requested."""
    calls: list[dict[str, Any]] = []

    def fake_process_video(*_: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"output_dir": tmp_path, "slides": [], "slides_md": tmp_path / "slides.md"}

    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("slidegeist.cli.process_video", fake_process_video)
    monkeypatch.setattr("slidegeist.cli.check_prerequisites", lambda: None)
    monkeypatch.setattr(
        "slidegeist.cli.resolve_video_path",
        lambda input_str, output_dir, cookies_from_browser=None: Path(input_str),
    )
    monkeypatch.setattr(sys, "argv", [
        "slidegeist", "process", str(tmp_path / "input.mp4"),
        "--transcriber", "voxtral", "--describer", "openrouter",
    ])

    cli.main()

    assert calls[0]["describer_provider"] == "openrouter"
    assert calls[0]["provider"] == "voxtral"


def test_missing_openrouter_key_stops_before_work(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Explicitly asking for the cloud describer without a key must fail loudly."""
    calls: list[dict[str, Any]] = []

    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("SLIDEGEIST_OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(
        "slidegeist.cli.process_video", lambda *a, **k: calls.append(k) or {}
    )
    monkeypatch.setattr("slidegeist.cli.check_prerequisites", lambda: None)
    monkeypatch.setattr(
        "slidegeist.cli.resolve_video_path",
        lambda input_str, output_dir, cookies_from_browser=None: Path(input_str),
    )
    monkeypatch.setattr(sys, "argv", [
        "slidegeist", "process", str(tmp_path / "input.mp4"), "--describer", "openrouter",
    ])

    with pytest.raises(SystemExit):
        cli.main()
    assert not calls
