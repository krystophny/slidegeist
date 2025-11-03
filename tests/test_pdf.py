"""Tests for PDF processing functionality."""

from pathlib import Path

import pytest

from slidegeist.pdf import extract_pdf_pages, is_pdf_file


def test_is_pdf_file_with_pdf(tmp_path: Path) -> None:
    """Test PDF file detection with actual PDF file."""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake pdf content")

    assert is_pdf_file(pdf_path)
    assert is_pdf_file(str(pdf_path))


def test_is_pdf_file_with_non_pdf(tmp_path: Path) -> None:
    """Test PDF file detection with non-PDF file."""
    video_path = tmp_path / "test.mp4"
    video_path.write_bytes(b"fake video content")

    assert not is_pdf_file(video_path)
    assert not is_pdf_file(str(video_path))


def test_is_pdf_file_with_nonexistent() -> None:
    """Test PDF file detection with nonexistent file."""
    assert not is_pdf_file(Path("/nonexistent/file.pdf"))


@pytest.mark.manual
def test_extract_pdf_pages_basic(tmp_path: Path) -> None:
    """Test basic PDF page extraction.

    This test requires PyMuPDF to be installed and a real PDF file.
    Mark as manual since it needs a real PDF fixture.
    """
    try:
        import fitz  # noqa: F401
    except ImportError:
        pytest.skip("PyMuPDF not installed")

    # Create a simple PDF with PyMuPDF
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((50, 50), "Page 1 Text")
    page2 = doc.new_page()
    page2.insert_text((50, 50), "Page 2 Text")

    pdf_path = tmp_path / "test.pdf"
    doc.save(pdf_path)
    doc.close()

    # Extract pages
    results = extract_pdf_pages(pdf_path, tmp_path, image_format="jpg", dpi=150)

    assert len(results) == 2

    # Check first page
    page_num, img_path, text = results[0]
    assert page_num == 1
    assert img_path.exists()
    assert img_path.suffix == ".jpg"
    assert img_path.name == "slide_001.jpg"
    assert "Page 1 Text" in text

    # Check second page
    page_num, img_path, text = results[1]
    assert page_num == 2
    assert img_path.exists()
    assert img_path.name == "slide_002.jpg"
    assert "Page 2 Text" in text


def test_extract_pdf_pages_missing_dependency(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test error handling when PyMuPDF is not installed."""
    # Mock missing fitz import
    import sys
    monkeypatch.setitem(sys.modules, "fitz", None)

    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    with pytest.raises(ImportError, match="PyMuPDF.*required"):
        extract_pdf_pages(pdf_path, tmp_path)
