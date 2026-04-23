"""Tesseract OCR for slide text extraction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class NoOpPipeline:
    """No-op OCR pipeline used when Tesseract is unavailable."""

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
            "engine": {"primary": None, "primary_version": None},
            "raw_text": "",
            "final_text": "",
            "blocks": [],
            "visual_elements": [],
            "model_response": "",
        }


class OcrPipeline:
    """Run Tesseract on a slide image and return structured text."""

    def __init__(self, primary_extractor: TesseractExtractor | None = None) -> None:
        self._primary = primary_extractor

    def process(
        self,
        image_path: Path,
        transcript_full_text: str,
        transcript_segments: list[dict[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "engine": {
                "primary": None,
                "primary_version": None,
                "describer": None,
                "describer_version": None,
            },
            "raw_text": "",
            "final_text": "",
            "blocks": [],
            "visual_elements": [],
            "model_response": None,
            "ai_description": "",
        }

        if self._primary is not None and self._primary.is_available:
            try:
                extracted = self._primary.extract(image_path)
                result.update({
                    "engine": extracted.get("engine", result["engine"]),
                    "raw_text": extracted.get("raw_text", ""),
                    "blocks": extracted.get("blocks", []),
                })
                result["final_text"] = result["raw_text"]
            except Exception as exc:  # pragma: no cover - defensive log path
                logger.warning("Tesseract OCR failed for %s: %s", image_path, exc)

        return result


class TesseractExtractor:
    """Wrap pytesseract calls, keeping dependency optional."""

    def __init__(self) -> None:
        try:
            import pytesseract
        except ImportError:
            self._pytesseract = None
            self._version = None
            self._available = False
            logger.info("pytesseract not installed; OCR will be disabled")
        else:
            self._pytesseract = pytesseract
            try:
                self._version = str(pytesseract.get_tesseract_version()).strip()
                self._available = True
            except Exception:
                self._version = None
                self._available = False
                self._pytesseract = None
                logger.info("tesseract binary not available; OCR will be disabled")

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def version(self) -> str | None:
        return self._version

    def extract(self, image_path: Path) -> dict[str, Any]:
        if not self.is_available:
            raise RuntimeError("pytesseract is not available")

        pytesseract = self._pytesseract
        assert pytesseract is not None

        from pytesseract import Output  # type: ignore[attr-defined]

        # PSM 1: automatic page segmentation with OSD — good for mixed slide layouts.
        data = pytesseract.image_to_data(
            str(image_path),
            lang="eng+deu",
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
            blocks.append({
                "text": text,
                "confidence": confidence,
                "bbox": [
                    int(data.get("left", [0])[idx]),
                    int(data.get("top", [0])[idx]),
                    int(data.get("width", [0])[idx]),
                    int(data.get("height", [0])[idx]),
                ],
                "level": int(data.get("level", [5])[idx]),
            })

        return {
            "engine": {
                "primary": "tesseract",
                "primary_version": self._version,
            },
            "raw_text": " ".join(raw_lines),
            "blocks": blocks,
        }


def build_default_ocr_pipeline() -> OcrPipeline:
    """Create the default Tesseract-backed OCR pipeline."""
    primary = TesseractExtractor()
    return OcrPipeline(primary_extractor=primary if primary.is_available else None)
