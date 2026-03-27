"""AI slide description generation through the local llama.cpp service."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from slidegeist.services import llama_cpp_complete

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Remove control artifacts and normalize spacing."""
    text = re.sub(r"\s+", " ", text)
    valid_chars = set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "äöüßÄÖÜ"
        "àâéèêëïîôùûüÿçÀÂÉÈÊËÏÎÔÙÛÜŸÇ"
        "0123456789"
        " .,;:!?()[]{}+-=*/<>|\\@#$%^&_~'\"`\n\t"
    )
    cleaned = "".join(c for c in text if c in valid_chars or c.isalpha() or c.isdigit())
    return cleaned.strip()


def get_system_instruction() -> str:
    """Return the system instruction used for slide reconstruction prompts."""
    return (
        "You reconstruct academic and scientific slides from text context. "
        "Prioritize completeness and accuracy over brevity. "
        "You do not have direct image access. "
        "Use only the supplied OCR and transcript context. "
        "If a detail cannot be recovered from context, write Unclear instead of inventing it."
    )


def get_user_prompt(transcript: str, ocr_text: str) -> str:
    """Build the user prompt for remote slide description generation."""
    context_parts = []

    if transcript.strip():
        context_parts.append(f"Speaker transcript: {transcript[:2000]}")

    if ocr_text.strip():
        context_parts.append(f"OCR text: {ocr_text[:4000]}")

    context = "\n".join(context_parts) if context_parts else "No context available"

    return f"""Describe this slide so another AI can reconstruct it from the available text context.

Reference context:
{context}

Output exactly 5 numbered sections:

1. TITLE
[Exact title text if available. Otherwise give a concise inferred title.]

2. TEXT CONTENT
[List visible text in reading order. Preserve bullets where possible.]

3. FORMULAS
[Write all formulas in LaTeX. If none, write "None".]

4. VISUAL ELEMENTS
[Describe diagrams, plots, tables, arrows, or figures if the context supports them. If unclear, write "Unclear".]

5. LAYOUT
[Describe overall structure and relative placement of content. If unclear, write "Unclear".]

END"""


class BaseSlideDescriber:
    """Interface for slide description backends."""

    name: str = "unknown"

    def describe(self, image_path: Path, transcript: str, ocr_text: str = "") -> str:
        raise NotImplementedError


class LlamaCppSlideDescriber(BaseSlideDescriber):
    """Slide describer backed by the local llama.cpp server."""

    def __init__(
        self,
        max_new_tokens: int = 1536,
        temperature: float = 0.0,
    ) -> None:
        self.name = "qwen3.5-9b (llama.cpp)"
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def describe(self, image_path: Path, transcript: str, ocr_text: str = "") -> str:
        del image_path

        if not transcript.strip() and not ocr_text.strip():
            logger.debug("Skipping AI description because no transcript or OCR text is available")
            return ""

        prompt = f"{get_system_instruction()}\n\n{get_user_prompt(transcript, ocr_text)}"
        response = llama_cpp_complete(
            prompt,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        return clean_text(response)


def build_ai_describer() -> BaseSlideDescriber:
    """Build the remote llama.cpp slide describer."""
    return LlamaCppSlideDescriber()
