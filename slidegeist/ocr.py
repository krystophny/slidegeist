"""OCR pipeline utilities for Tesseract-based slide text extraction."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RefinementOutput:
    """Structured response from an OCR refinement model."""

    text: str
    visual_elements: list[str]
    raw_response: str


class BaseRefiner:
    """Interface for optional OCR refinement models."""

    name: str = "unknown"
    version: str | None = None

    def is_available(self) -> bool:  # pragma: no cover - interface hook
        return False

    def refine(
        self,
        image_path: Path,
        raw_text: str,
        transcript: str,
        segments: list[dict[str, Any]],
    ) -> RefinementOutput | None:  # pragma: no cover - interface hook
        raise NotImplementedError


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
                "refiner": None,
                "refiner_version": None,
            },
            "raw_text": "",
            "final_text": "",
            "blocks": [],
            "visual_elements": [],
            "model_response": "",
        }


class OcrPipeline:
    """Compose Tesseract OCR with optional refinement."""

    def __init__(
        self,
        primary_extractor: TesseractExtractor | None = None,
        refiner: BaseRefiner | None = None,
    ) -> None:
        self._primary = primary_extractor
        self._refiner = refiner

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
                "refiner": None,
                "refiner_version": None,
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
                result = self._primary.extract(image_path)
                raw_payload.update({
                    "engine": result.get("engine", raw_payload["engine"]),
                    "raw_text": result.get("raw_text", ""),
                    "blocks": result.get("blocks", []),
                })
                raw_payload["final_text"] = raw_payload["raw_text"]
            except Exception as exc:  # pragma: no cover - defensive log path
                logger.warning("Primary OCR failed for %s: %s", image_path, exc)

        raw_text = raw_payload["raw_text"]

        if self._refiner is not None and self._refiner.is_available():
            try:
                refined = self._refiner.refine(
                    image_path=image_path,
                    raw_text=raw_text,
                    transcript=transcript_full_text,
                    segments=transcript_segments,
                )
            except Exception as exc:  # pragma: no cover - defensive log path
                logger.warning("OCR refinement failed for %s: %s", image_path, exc)
            else:
                if refined is not None:
                    if refined.text:
                        raw_payload["final_text"] = refined.text.strip()
                    if refined.visual_elements:
                        raw_payload["visual_elements"] = [str(item) for item in refined.visual_elements]
                    raw_payload["model_response"] = refined.raw_response
                    raw_payload["engine"]["refiner"] = self._refiner.name
                    raw_payload["engine"]["refiner_version"] = self._refiner.version

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

        pytesseract = self._pytesseract  # type: ignore[assignment]
        assert pytesseract is not None

        from pytesseract import Output  # type: ignore[attr-defined]

        # Use both English and German for multilingual support
        # PSM 1 = Automatic page segmentation with OSD (orientation and script detection)
        # Best for slides with multiple blocks and varying orientations
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
                "refiner": None,
                "refiner_version": None,
            },
            "raw_text": raw_text,
            "blocks": blocks,
        }
def _parse_model_response(response: str, fallback_text: str) -> RefinementOutput:
    """Parse model response JSON, tolerating minor formatting errors."""

    response = response.strip()
    candidate = response

    if not candidate:
        return RefinementOutput(text=fallback_text, visual_elements=[], raw_response=response)

    json_obj: dict[str, Any] | None = None
    if candidate.startswith("{"):
        try:
            json_obj = json.loads(candidate)
        except json.JSONDecodeError:
            pass

    if json_obj is None:
        try:
            start = candidate.index("{")
            end = candidate.rfind("}")
            json_obj = json.loads(candidate[start : end + 1])
        except Exception:
            return RefinementOutput(text=candidate, visual_elements=[], raw_response=response)

    text_value = str(json_obj.get("text", "")).strip()
    elements_raw = json_obj.get("visual_elements", [])
    if isinstance(elements_raw, str):
        elements_list = [elements_raw]
    elif isinstance(elements_raw, list):
        elements_list = [str(item).strip() for item in elements_raw if str(item).strip()]
    else:
        elements_list = []

    if not text_value:
        text_value = candidate

    return RefinementOutput(text=text_value, visual_elements=elements_list, raw_response=response)


def build_default_ocr_pipeline() -> OcrPipeline:
    """Create the default OCR pipeline with Tesseract only (no refinement)."""
    primary = TesseractExtractor()

    if not primary.is_available:
        primary = None  # type: ignore[assignment]

    return OcrPipeline(primary_extractor=primary, refiner=None)
