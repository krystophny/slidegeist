"""PDF processing for slide extraction."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def is_pdf_file(path: Path | str) -> bool:
    """Check if path points to a PDF file.

    Args:
        path: Path to check.

    Returns:
        True if path exists and is a PDF file.
    """
    path_obj = Path(path) if isinstance(path, str) else path
    return path_obj.exists() and path_obj.suffix.lower() == ".pdf"


def extract_pdf_pages(
    pdf_path: Path,
    output_dir: Path,
    image_format: str = "jpg",
    dpi: int = 300,
) -> list[tuple[int, Path, str]]:
    """Extract pages from PDF as images with embedded text.

    Args:
        pdf_path: Path to PDF file.
        output_dir: Directory to save extracted images.
        image_format: Output image format ('jpg' or 'png').
        dpi: DPI for image rendering (300 for good quality).

    Returns:
        List of (page_number, image_path, embedded_text) tuples.

    Raises:
        ImportError: If PyMuPDF (fitz) is not installed.
        Exception: If PDF extraction fails.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "PyMuPDF (fitz) is required for PDF support. "
            "Install with: pip install pymupdf"
        )

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    slides_dir = output_dir / "slides"
    slides_dir.mkdir(parents=True, exist_ok=True)

    results: list[tuple[int, Path, str]] = []

    logger.info(f"Extracting pages from PDF: {pdf_path}")

    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        logger.info(f"PDF has {total_pages} pages")

        for page_num in range(total_pages):
            page = doc[page_num]

            # Extract embedded text
            text = page.get_text().strip()

            # Render page as image
            mat = fitz.Matrix(dpi / 72, dpi / 72)  # Scale matrix for DPI
            pix = page.get_pixmap(matrix=mat)

            # Save image
            slide_filename = f"slide_{page_num + 1:03d}.{image_format}"
            image_path = slides_dir / slide_filename

            if image_format == "png":
                pix.save(image_path)
            else:  # jpg
                # Convert to RGB for JPEG (remove alpha channel)
                if pix.alpha:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                pix.save(image_path, "jpeg", jpg_quality=95)

            results.append((page_num + 1, image_path, text))
            logger.debug(f"Extracted page {page_num + 1}/{total_pages}: {image_path}")

        doc.close()
        logger.info(f"Extracted {len(results)} pages to {slides_dir}")

    except Exception as e:
        raise Exception(f"Failed to extract PDF pages: {e}")

    return results
