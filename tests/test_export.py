"""Tests for Markdown export functionality."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from slidegeist.export import (
    _parse_existing_markdown,
    _save_incremental_ai_description,
    export_slides_json,
    run_ai_descriptions,
)
from slidegeist.ocr import build_default_ocr_pipeline
from slidegeist.transcribe import Segment


def _make_image(path: Path, color: int) -> None:
    """Create a simple solid color test image."""
    matrix = np.full((10, 20, 3), color, dtype=np.uint8)
    cv2.imwrite(str(path), matrix)


def _make_text_image(path: Path, text: str) -> None:
    """Create an image with readable text for OCR testing."""
    # Create white background
    img = np.full((100, 400, 3), 255, dtype=np.uint8)

    # Add black text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (10, 50), font, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imwrite(str(path), img)


def test_export_slides_manifest_and_payloads(tmp_path: Path) -> None:
    video_path = Path("/fake/video.mp4")

    slides_dir = tmp_path / "slides"
    slides_dir.mkdir()
    img1 = slides_dir / "slide_001.jpg"
    img2 = slides_dir / "slide_002.jpg"
    _make_text_image(img1, "QUANTUM PHYSICS")
    _make_text_image(img2, "NEWTON LAWS")

    slide_metadata = [
        (1, 0.0, 10.0, img1),
        (2, 10.0, 20.0, img2),
    ]

    transcript_segments: list[Segment] = [
        {"start": 0.0, "end": 5.0, "text": "Welcome to the lecture.", "words": []},
        {"start": 5.0, "end": 10.0, "text": "Today we discuss physics.", "words": []},
        {"start": 10.0, "end": 15.0, "text": "Let's start with Newton.", "words": []},
        {"start": 15.0, "end": 20.0, "text": "And then Einstein.", "words": []},
    ]

    output_file = tmp_path / "index.md"
    ocr_pipeline = build_default_ocr_pipeline()

    export_slides_json(
        video_path,
        slide_metadata,
        transcript_segments,
        output_file,
        "tiny",
        ocr_pipeline=ocr_pipeline,
        split_slides=True,
    )

    assert output_file.exists()

    index_content = output_file.read_text()
    assert "# Lecture Slides" in index_content
    assert "video.mp4" in index_content
    assert "tiny" in index_content
    assert "Slide 1" in index_content
    assert "Slide 2" in index_content

    # Check per-slide markdown (in output root, not slides/)
    slide1_md = tmp_path / "slide_001.md"
    assert slide1_md.exists()
    slide1_content = slide1_md.read_text()
    assert "---" in slide1_content
    assert "id: slide_001" in slide1_content
    assert "index: 1" in slide1_content
    assert "time_start: 0.0" in slide1_content
    assert "time_end: 10.0" in slide1_content
    assert "# Slide 1" in slide1_content
    assert "Welcome to the lecture." in slide1_content

    # Check OCR content (if Tesseract available, should have content)
    if ocr_pipeline._primary is not None and ocr_pipeline._primary.is_available:
        assert "## OCR Text" in slide1_content
        # Should extract some text from the image
        assert "QUANTUM" in slide1_content or "quantum" in slide1_content.lower()


def test_export_slides_handles_empty_transcript(tmp_path: Path) -> None:
    video_path = Path("/fake/video.mp4")
    slides_dir = tmp_path / "slides"
    slides_dir.mkdir()
    img1 = slides_dir / "slide_001.jpg"
    _make_text_image(img1, "SUMMARY")

    slide_metadata = [
        (1, 0.0, 10.0, img1),
    ]

    transcript_segments: list[Segment] = []

    output_file = tmp_path / "index.md"
    ocr_pipeline = build_default_ocr_pipeline()

    export_slides_json(
        video_path,
        slide_metadata,
        transcript_segments,
        output_file,
        "base",
        ocr_pipeline=ocr_pipeline,
        split_slides=True,
    )

    slide1_md = tmp_path / "slide_001.md"
    assert slide1_md.exists()
    content = slide1_md.read_text()

    # With empty transcript, no transcript section
    assert "## Transcript" not in content

    # Should still have OCR content if Tesseract available
    if ocr_pipeline._primary is not None and ocr_pipeline._primary.is_available:
        assert "## OCR Text" in content


def test_export_slides_empty_metadata(tmp_path: Path) -> None:
    video_path = Path("/fake/video.mp4")
    slide_metadata: list[tuple[int, float, float, Path]] = []
    transcript_segments: list[Segment] = []

    output_file = tmp_path / "index.md"
    ocr_pipeline = build_default_ocr_pipeline()

    export_slides_json(
        video_path,
        slide_metadata,
        transcript_segments,
        output_file,
        "tiny",
        ocr_pipeline=ocr_pipeline,
        split_slides=True,
    )

    assert output_file.exists()
    content = output_file.read_text()
    assert "# Lecture Slides" in content
    assert "video.mp4" in content
    # No slide markdown files in root (split mode with no slides)
    md_files = list(tmp_path.glob("slide_*.md"))
    assert len(md_files) == 0


def test_checkpoint_export_does_not_invoke_ocr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    slides = tmp_path / "slides"
    slides.mkdir()
    image = slides / "slide_001.jpg"
    _make_image(image, 255)

    def unexpected_ocr_build() -> None:
        pytest.fail("checkpoint export invoked OCR")

    monkeypatch.setattr("slidegeist.ocr.build_default_ocr_pipeline", unexpected_ocr_build)

    output = tmp_path / "slides.md"
    export_slides_json(
        Path("/known/video.mp4"),
        [(1, 0.0, 10.0, image)],
        [],
        output,
        run_ocr=False,
    )

    assert "slides/slide_001.jpg" in output.read_text()
    assert "OCR Text" not in output.read_text()


def test_ai_description_reuses_persisted_ocr(
    tmp_path: Path,
) -> None:
    slides = tmp_path / "slides"
    slides.mkdir()
    image = slides / "slide_001.jpg"
    _make_image(image, 255)
    output = tmp_path / "slides.md"
    output.write_text(
        '<a name="slide_001"></a>\n'
        "## Slide 1\n\n"
        "[![Slide](slides/slide_001.jpg)](slides/slide_001.jpg)\n\n"
        "### OCR Text\n\nPersisted equation text\n\n---\n",
        encoding="utf-8",
    )

    class FailingOcr:
        def process(self, *_: object, **__: object) -> dict[str, object]:
            pytest.fail("AI description repeated persisted OCR")

    class RecordingDescriber:
        seen_ocr = ""

        def describe(self, _: Path, __: str, ocr_text: str) -> str:
            self.seen_ocr = ocr_text
            return "0. FRAME TYPE\nSLIDE\n\n1. TITLE\nKnown state"

    describer = RecordingDescriber()
    descriptions = run_ai_descriptions(
        [(1, 0.0, 10.0, image)],
        [],
        describer,  # type: ignore[arg-type]
        FailingOcr(),  # type: ignore[arg-type]
        output_path=output,
    )

    assert describer.seen_ocr == "Persisted equation text"
    assert descriptions["slide_001"].startswith("0. FRAME TYPE\nSLIDE")


def test_split_description_checkpoint_is_parsed_and_replaced(
    tmp_path: Path,
) -> None:
    index = tmp_path / "index.md"
    index.write_text("# Lecture Slides\n", encoding="utf-8")
    slide = tmp_path / "slide_001.md"
    slide.write_text(
        "---\nid: slide_001\nindex: 1\n---\n\n"
        "# Slide 1\n\n"
        "## Transcript\n\nKnown speech\n\n"
        "## AI Description (for reconstruction)\n\n"
        "0. FRAME TYPE\nSLIDE\n\n1. TITLE\nOld title\n",
        encoding="utf-8",
    )

    parsed = _parse_existing_markdown(index)
    assert parsed["slide_001"]["transcript"] == "Known speech"
    assert "Old title" in parsed["slide_001"]["ai_description"]

    _save_incremental_ai_description(
        index,
        "slide_001",
        "0. FRAME TYPE\nSLIDE\n\n1. TITLE\nNew title",
        parsed,
    )

    updated = slide.read_text(encoding="utf-8")
    assert "New title" in updated
    assert "Old title" not in updated
    assert updated.count("## AI Description (for reconstruction)") == 1


def test_combined_checkpoint_repair_removes_duplicate_failed_descriptions(
    tmp_path: Path,
) -> None:
    output = tmp_path / "slides.md"
    output.write_text(
        '<a name="slide_001"></a>\n'
        "## Slide 1\n\n"
        "### OCR Text\n\nCONTENTS\n\n"
        "---\n\n"
        "### AI Description (for reconstruction)\n\nFirst truncated response\n\n"
        "### AI Description (for reconstruction)\n\nSecond truncated response\n\n"
        '<a name="slide_002"></a>\n'
        "## Slide 2\n\n### OCR Text\n\nKeep this slide\n\n---\n",
        encoding="utf-8",
    )

    _save_incremental_ai_description(
        output,
        "slide_001",
        "0. FRAME TYPE\nSLIDE\n\n1. TITLE\nRecovered",
        _parse_existing_markdown(output),
    )

    updated = output.read_text(encoding="utf-8")
    first_slide, second_slide = updated.split('<a name="slide_002"></a>')
    assert first_slide.count("### AI Description (for reconstruction)") == 1
    assert "Recovered" in first_slide
    assert "truncated response" not in first_slide
    assert "Keep this slide" in second_slide


def test_single_speaker_transcript_text_is_unprefixed() -> None:
    """Single-speaker windows must render exactly as before speaker support."""
    from slidegeist.export import _collect_transcript_text

    segments = [
        {"start": 0.0, "end": 1.0, "text": "Hello there", "words": [], "speaker": "SPEAKER_00"},
        {"start": 1.0, "end": 2.0, "text": "and welcome", "words": [], "speaker": "SPEAKER_00"},
    ]

    assert _collect_transcript_text(segments, 0.0, 5.0) == "Hello there and welcome"


def test_multi_speaker_slide_window_prefixes_speakers() -> None:
    from slidegeist.export import _collect_transcript_text

    segments = [
        {"start": 0.0, "end": 1.0, "text": "What is the closure?", "words": [], "speaker": "SPEAKER_01"},
        {"start": 1.0, "end": 2.0, "text": "Good question.", "words": [], "speaker": "SPEAKER_00"},
        {"start": 2.0, "end": 3.0, "text": "It truncates the hierarchy.", "words": [], "speaker": "SPEAKER_00"},
    ]

    rendered = _collect_transcript_text(segments, 0.0, 5.0)

    assert "**SPEAKER_01:** What is the closure?" in rendered
    # Consecutive runs by the same speaker are merged, not repeated per segment.
    assert "**SPEAKER_00:** Good question. It truncates the hierarchy." in rendered
    # Must not emit anything that terminates a markdown transcript section.
    for line in rendered.splitlines():
        assert not line.startswith("#")
        assert line.strip() != "---"


def test_local_describer_is_never_parallelised(monkeypatch) -> None:
    """The local llama.cpp server has one slot; concurrency would only queue."""
    from slidegeist.export import get_describe_concurrency

    class Local:
        provider = "local"
        name = "gemma-4-26b-a4b (llama.cpp)"

    class Remote:
        provider = "openrouter"
        name = "google/gemma-4-26b-a4b-it (OpenRouter)"

    monkeypatch.setenv("SLIDEGEIST_DESCRIBE_CONCURRENCY", "16")
    assert get_describe_concurrency(Local()) == 1, "local must stay sequential"
    assert get_describe_concurrency(Remote()) == 16

    monkeypatch.delenv("SLIDEGEIST_DESCRIBE_CONCURRENCY")
    assert get_describe_concurrency(Remote()) == 8


def test_concurrent_descriptions_are_keyed_by_slide_not_order(tmp_path) -> None:
    """Out-of-order completion must not misattribute descriptions."""
    import time

    from slidegeist.export import run_ai_descriptions

    class SlowFirstDescriber:
        provider = "openrouter"
        name = "fake (OpenRouter)"

        def describe(self, image_path, transcript, ocr_text=""):
            # slide_001 finishes last despite being submitted first
            if image_path.stem == "slide_001":
                time.sleep(0.2)
            return (
                f"0. FRAME TYPE\nSLIDE\n\n1. TITLE\n{image_path.stem}\n\n"
                "2. TEXT CONTENT\nx\n\n3. FORMULAS\nNone\n\n"
                "4. VISUAL ELEMENTS\nNone\n\n5. LAYOUT\nOne column"
            )

    meta = []
    for index in (1, 2, 3):
        path = tmp_path / f"slide_{index:03d}.jpg"
        path.write_bytes(b"x")
        meta.append((index, float(index), float(index + 1), path))

    result = run_ai_descriptions(
        slide_metadata=meta,
        transcript_segments=[],
        describer=SlowFirstDescriber(),
        ocr_pipeline=None,
        output_path=None,
    )

    assert set(result) == {"slide_001", "slide_002", "slide_003"}
    for slide_id in result:
        assert f"1. TITLE\n{slide_id}" in result[slide_id], "description bound to wrong slide"
