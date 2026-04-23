"""Tests for OCR helpers and manual end-to-end checks."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from slidegeist.ocr import build_default_ocr_pipeline


def test_tesseract_ocr_pipeline(tmp_path: Path) -> None:
    """Test Tesseract OCR with a generated text image."""
    # Create image with readable text
    img = np.full((200, 600, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "HELLO WORLD", (50, 100), font, 2.0, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, "Test 123", (50, 150), font, 1.5, (0, 0, 0), 2, cv2.LINE_AA)

    image_path = tmp_path / "test_slide.jpg"
    cv2.imwrite(str(image_path), img)

    pipeline = build_default_ocr_pipeline()

    # Test with or without Tesseract
    if pipeline._primary is not None and pipeline._primary.is_available:
        result = pipeline.process(
            image_path=image_path,
            transcript_full_text="Hello world test",
            transcript_segments=[{"start": 0.0, "end": 5.0, "text": "Hello world test"}],
        )

        # Should extract some text
        assert result["final_text"], "OCR should extract text from image"
        assert "HELLO" in result["final_text"] or "Hello" in result["final_text"].lower()
        assert result["engine"]["primary"] == "tesseract"
    else:
        # Graceful fallback when Tesseract not installed
        result = pipeline.process(
            image_path=image_path,
            transcript_full_text="Hello world test",
            transcript_segments=[{"start": 0.0, "end": 5.0, "text": "Hello world test"}],
        )
        assert result["final_text"] == ""
        assert result["raw_text"] == ""


@pytest.mark.manual
def test_manual_ocr_pipeline(tmp_path: Path) -> None:
    from PIL import Image, ImageDraw

    from slidegeist.ocr import build_default_ocr_pipeline

    image_path = tmp_path / "slide_manual.jpg"
    image = Image.new("RGB", (640, 360), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((50, 200, 200, 280), outline="red", width=8)
    draw.text((60, 60), "Deep Learning Overview", fill=(0, 0, 0))
    draw.text((60, 120), "- Convolutional Networks", fill=(0, 0, 0))
    image.save(image_path)

    pipeline = build_default_ocr_pipeline()
    assert pipeline._primary is not None and pipeline._primary.is_available, "Tesseract missing"

    result = pipeline.process(
        image_path=image_path,
        transcript_full_text="Today we cover convolutional networks",
        transcript_segments=[{"start": 0.0, "end": 5.0, "text": "Today we cover convolutional networks"}],
    )

    assert "Deep" in result["final_text"]
    assert result["visual_elements"] == []
