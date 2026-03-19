"""OCR pipeline utilities using Tesseract for text extraction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class NoOpPipeline:
    """No-op OCR pipeline that returns empty results."""

    class _NoOpPrimary:
        is_available = False

    _primary = _NoOpPrimary()

    def process(
        self,
        image_path: Path,
        transcript_full_text: str,
        transcript_segments: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "engine": {
                "primary": None,
                "primary_version": None,
            },
            "raw_text": "",
            "final_text": "",
            "blocks": [],
            "visual_elements": [],
        }


class OcrPipeline:
    """Tesseract OCR pipeline."""

    def __init__(
        self,
        primary_extractor: TesseractExtractor | None = None,
    ) -> None:
        self._primary = primary_extractor

    def process(
        self,
        image_path: Path,
        transcript_full_text: str,
        transcript_segments: list[dict[str, Any]],
    ) -> dict[str, Any]:
        raw_payload: dict[str, Any] = {
            "engine": {
                "primary": None,
                "primary_version": None,
            },
            "raw_text": "",
            "final_text": "",
            "blocks": [],
            "visual_elements": [],
        }

        if self._primary is not None and self._primary.is_available:
            try:
                result = self._primary.extract(image_path)
                raw_payload.update({
                    "engine": result.get("engine", raw_payload["engine"]),
                    "raw_text": result.get("raw_text", ""),
                    "blocks": result.get("blocks", []),
                })
                raw_payload["final_text"] = raw_payload["raw_text"]
            except Exception as exc:  # pragma: no cover - defensive log path
                logger.warning("Primary OCR failed for %s: %s", image_path, exc)

        if not raw_payload["final_text"]:
            raw_payload["final_text"] = raw_payload["raw_text"]

        return raw_payload


class TesseractExtractor:
    """Wrap pytesseract calls, keeping dependency optional."""

    def __init__(self) -> None:
        try:
            import pytesseract
        except ImportError:
            self._pytesseract = None
            self._version = None
            logger.info("pytesseract not installed; OCR will be disabled")
        else:
            self._pytesseract = pytesseract
            try:
                self._version = str(pytesseract.get_tesseract_version()).strip()
            except Exception:
                self._pytesseract = None
                self._version = None
                logger.info("tesseract binary not found in PATH; OCR will be disabled")

    @property
    def is_available(self) -> bool:
        return self._pytesseract is not None

    @property
    def version(self) -> str | None:
        return self._version

    def extract(self, image_path: Path) -> dict[str, Any]:
        if not self.is_available:
            raise RuntimeError("pytesseract is not available")

        pytesseract = self._pytesseract  # type: ignore[assignment]
        assert pytesseract is not None

        from pytesseract import Output  # type: ignore[attr-defined]

        # Use both English and German for multilingual support
        # PSM 1 = Automatic page segmentation with OSD
        data = pytesseract.image_to_data(
            str(image_path),
            lang='eng+deu',
            output_type=Output.DICT,
            config="--psm 1",
        )

        blocks: list[dict[str, Any]] = []
        raw_lines: list[str] = []

        for idx, text in enumerate(data.get("text", [])):
            text = text.strip()
            conf_str = data.get("conf", ["-1"])[idx]
            try:
                confidence = float(conf_str)
            except ValueError:
                confidence = -1.0

            if not text:
                continue

            raw_lines.append(text)
            block = {
                "text": text,
                "confidence": confidence,
                "bbox": [
                    int(data.get("left", [0])[idx]),
                    int(data.get("top", [0])[idx]),
                    int(data.get("width", [0])[idx]),
                    int(data.get("height", [0])[idx]),
                ],
                "level": int(data.get("level", [5])[idx]),
            }
            blocks.append(block)

        raw_text = " ".join(raw_lines)

        return {
            "engine": {
                "primary": "tesseract",
                "primary_version": self._version,
            },
            "raw_text": raw_text,
            "blocks": blocks,
        }


def build_default_ocr_pipeline() -> OcrPipeline:
    """Create the default OCR pipeline with Tesseract."""
    primary = TesseractExtractor()

    if not primary.is_available:
        primary = None  # type: ignore[assignment]

    return OcrPipeline(primary_extractor=primary)
