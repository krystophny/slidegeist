"""Tests for Markdown export functionality."""

from pathlib import Path

import cv2
import numpy as np

from slidegeist.export import export_slides_json
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


def test_export_pdf_hides_timestamps_combined_mode(tmp_path: Path) -> None:
    """Test PDF export hides timestamps in combined mode."""
    pdf_path = Path("/fake/document.pdf")

    slides_dir = tmp_path / "slides"
    slides_dir.mkdir()
    img1 = slides_dir / "slide_001.jpg"
    img2 = slides_dir / "slide_002.jpg"
    _make_text_image(img1, "INTRO")
    _make_text_image(img2, "SUMMARY")

    slide_metadata = [
        (1, 0.0, 0.0, img1),  # Zero timestamps for PDFs
        (2, 0.0, 0.0, img2),
    ]

    transcript_segments: list[Segment] = []  # No transcript for PDFs

    output_file = tmp_path / "slides.md"
    ocr_pipeline = build_default_ocr_pipeline()

    pdf_texts = {
        "slide_001": "Introduction to the topic",
        "slide_002": "Summary and conclusions",
    }

    export_slides_json(
        pdf_path,
        slide_metadata,
        transcript_segments,
        output_file,
        "",  # No model for PDFs
        ocr_pipeline=ocr_pipeline,
        split_slides=False,
        pdf_texts=pdf_texts,
    )

    assert output_file.exists()
    content = output_file.read_text()

    # Check header uses PDF label
    assert "**PDF:** document.pdf" in content
    assert "**Video:**" not in content

    # Check no duration or transcription model
    assert "**Duration:**" not in content
    assert "**Transcription Model:**" not in content

    # Check no timestamps in slide sections
    assert "**Time:**" not in content
    assert "00:00 - 00:00" not in content

    # Check PDF embedded text sections exist
    assert "### PDF Embedded Text" in content
    assert "Introduction to the topic" in content
    assert "Summary and conclusions" in content

    # Check no transcript sections
    assert "### Transcript" not in content


def test_export_pdf_hides_timestamps_split_mode(tmp_path: Path) -> None:
    """Test PDF export hides timestamps in split mode."""
    pdf_path = Path("/fake/document.pdf")

    slides_dir = tmp_path / "slides"
    slides_dir.mkdir()
    img1 = slides_dir / "slide_001.jpg"
    _make_text_image(img1, "TITLE")

    slide_metadata = [
        (1, 0.0, 0.0, img1),  # Zero timestamps for PDFs
    ]

    transcript_segments: list[Segment] = []

    output_file = tmp_path / "index.md"
    ocr_pipeline = build_default_ocr_pipeline()

    pdf_texts = {
        "slide_001": "Page 1 embedded text",
    }

    export_slides_json(
        pdf_path,
        slide_metadata,
        transcript_segments,
        output_file,
        "",
        ocr_pipeline=ocr_pipeline,
        split_slides=True,
        pdf_texts=pdf_texts,
    )

    # Check index file
    assert output_file.exists()
    index_content = output_file.read_text()

    assert "**PDF:** document.pdf" in index_content
    assert "**Duration:**" not in index_content
    assert "**Transcription Model:**" not in index_content

    # Check index link has no timestamp
    assert "[Slide 1](slide_001.md)" in index_content
    # Should not have " • 00:00-00:00" suffix
    lines_with_slide1 = [line for line in index_content.split("\n") if "Slide 1" in line]
    assert len(lines_with_slide1) == 1
    assert " • 00:00-00:00" not in lines_with_slide1[0]

    # Check individual slide file
    slide1_md = tmp_path / "slide_001.md"
    assert slide1_md.exists()
    slide1_content = slide1_md.read_text()

    # Should not have time fields in front matter
    assert "time_start:" not in slide1_content
    assert "time_end:" not in slide1_content

    # Should have PDF text section
    assert "## PDF Embedded Text" in slide1_content
    assert "Page 1 embedded text" in slide1_content

    # Should not have transcript section
    assert "## Transcript" not in slide1_content


def test_export_video_shows_timestamps(tmp_path: Path) -> None:
    """Test video export shows timestamps (regression test)."""
    video_path = Path("/fake/video.mp4")

    slides_dir = tmp_path / "slides"
    slides_dir.mkdir()
    img1 = slides_dir / "slide_001.jpg"
    _make_text_image(img1, "INTRO")

    slide_metadata = [
        (1, 0.0, 10.5, img1),  # Nonzero timestamps for videos
    ]

    transcript_segments: list[Segment] = [
        {"start": 0.0, "end": 5.0, "text": "Hello world", "words": []},
    ]

    output_file = tmp_path / "slides.md"
    ocr_pipeline = build_default_ocr_pipeline()

    export_slides_json(
        video_path,
        slide_metadata,
        transcript_segments,
        output_file,
        "tiny",
        ocr_pipeline=ocr_pipeline,
        split_slides=False,
    )

    content = output_file.read_text()

    # Should have video label
    assert "**Video:** video.mp4" in content

    # Should have duration and model
    assert "**Duration:**" in content
    assert "**Transcription Model:** tiny" in content

    # Should have timestamps in slide section
    assert "**Time:** 00:00 - 00:10" in content

    # Should have transcript section
    assert "### Transcript" in content
    assert "Hello world" in content
