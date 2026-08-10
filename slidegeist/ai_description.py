"""AI slide description generation through the local llama.cpp service."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from slidegeist.constants import DEFAULT_DESCRIBER
from slidegeist.services import (
    get_llama_cpp_max_tokens,
    get_llama_cpp_model,
    get_llama_cpp_model_override,
    get_openrouter_model,
    llama_cpp_complete,
    openrouter_describe,
)

logger = logging.getLogger(__name__)


def compact_decorative_leaders(text: str) -> str:
    """Replace OCR dot leaders with a compact semantic marker."""
    return re.sub(r"(?:\.\s*){3,}", "[leader] ", text)


def clean_text(text: str) -> str:
    """Remove control artifacts and normalize spacing."""
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    valid_chars = set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "äöüßÄÖÜ"
        "àâéèêëïîôùûüÿçÀÂÉÈÊËÏÎÔÙÛÜŸÇ"
        "0123456789"
        " .,;:!?()[]{}+-=*/<>|\\@#$%^&_~'\"`•\n\t"
    )
    cleaned = "".join(c for c in text if c in valid_chars or c.isalpha() or c.isdigit())
    return cleaned.strip()


def get_system_instruction() -> str:
    """Return the system instruction used for slide reconstruction prompts."""
    return (
        "You reconstruct academic and scientific slides from the image and text context. "
        "Be concise without omitting visible content needed for reconstruction. "
        "Treat the image as primary visual evidence and OCR/transcript as supporting context. "
        "If a detail cannot be recovered from context, write Unclear instead of inventing it."
    )


def get_user_prompt(transcript: str, ocr_text: str) -> str:
    """Build the user prompt for remote slide description generation."""
    context_parts = []

    if transcript.strip():
        context_parts.append(f"Speaker transcript: {transcript[:2000]}")

    if ocr_text.strip():
        compact_ocr = compact_decorative_leaders(ocr_text)
        context_parts.append(f"OCR text: {compact_ocr[:4000]}")

    context = "\n".join(context_parts) if context_parts else "No context available"

    return f"""Classify this frame, then describe it so another AI can reconstruct it.

Reference context:
{context}

Start with this section:

0. FRAME TYPE
[Write exactly "SLIDE" for a lecture slide, handwritten teaching page, whiteboard, or other frame whose main content is instructional. Write exactly "NON-SLIDE" for a desktop, file browser, document chooser, menu, dialog, recording control, loading screen, blank screen, or a teaching page substantially obscured by operating-system UI.]

If the frame type is NON-SLIDE, stop immediately after the word NON-SLIDE.
If the frame type is SLIDE, continue with all five reconstruction sections:
Use compact fragments, quote each visible item once, and do not repeat TEXT CONTENT in
VISUAL ELEMENTS or LAYOUT. Aim for at most 200 words unless additional formulas require more.
Represent each run of decorative dot leaders or repeated punctuation once as "[leader]";
never transcribe the individual repeated characters.

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
    """Multimodal slide describer backed by an OpenAI-compatible llama.cpp server."""

    def __init__(
        self,
        max_new_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> None:
        model_id = get_llama_cpp_model_override() or get_llama_cpp_model() or "unknown"
        self.name = f"{model_id} (llama.cpp)"
        self.max_new_tokens = (
            get_llama_cpp_max_tokens() if max_new_tokens is None else max_new_tokens
        )
        self.temperature = temperature

    def describe(self, image_path: Path, transcript: str, ocr_text: str = "") -> str:
        prompt = f"{get_system_instruction()}\n\n{get_user_prompt(transcript, ocr_text)}"
        response = llama_cpp_complete(
            prompt,
            image_path=image_path,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        description = clean_text(response)

        from slidegeist.frame_filter import is_complete_frame_description

        if is_complete_frame_description(description):
            return description

        repaired = llama_cpp_complete(
            _repair_prompt(description, transcript, ocr_text),
            image_path=image_path,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        return clean_text(repaired)


class OpenRouterSlideDescriber(BaseSlideDescriber):
    """Multimodal slide describer backed by OpenRouter (Gemma 4 by default)."""

    def __init__(
        self,
        model: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.model = model or get_openrouter_model()
        self.name = f"{self.model} (OpenRouter)"
        self.max_new_tokens = (
            get_llama_cpp_max_tokens() if max_new_tokens is None else max_new_tokens
        )
        self.temperature = temperature

    def describe(self, image_path: Path, transcript: str, ocr_text: str = "") -> str:
        prompt = f"{get_system_instruction()}\n\n{get_user_prompt(transcript, ocr_text)}"
        description = clean_text(
            openrouter_describe(
                prompt,
                image_path=image_path,
                model=self.model,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
        )

        from slidegeist.frame_filter import is_complete_frame_description

        if is_complete_frame_description(description):
            return description

        repaired = openrouter_describe(
            _repair_prompt(description, transcript, ocr_text),
            image_path=image_path,
            model=self.model,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        return clean_text(repaired)


def _repair_prompt(description: str, transcript: str, ocr_text: str) -> str:
    """Build the retry prompt for an incomplete first attempt.

    The retry is always re-sent *with the image*. An earlier version omitted it
    and asked the model to reconstruct the page from transcript and OCR alone,
    which invites invented formulas on exactly the handwritten pages where the
    first attempt struggled.
    """
    compact_attempt = compact_decorative_leaders(description)[:4000]
    return (
        f"{get_system_instruction()}\n\n"
        "The previous attempt below was incomplete or repetitive. Look at the image again "
        "and rewrite it as one complete frame classification. Transcribe only what is "
        "visible; never invent content that the image does not show. Keep the result under "
        "150 words and include every required numbered section for a SLIDE. Never emit dot "
        "leaders or repeated punctuation.\n\n"
        f"{get_user_prompt(transcript, ocr_text)}\n\n"
        f"Incomplete attempt:\n{compact_attempt}"
    )


def build_ai_describer(provider: str = DEFAULT_DESCRIBER) -> BaseSlideDescriber:
    """Build a slide describer for the requested provider.

    OpenRouter (Gemma 4) is the default; ``local`` uses the llama.cpp server.
    There is no automatic fallback between them.
    """
    if provider in ("openrouter", "gemma", "remote"):
        return OpenRouterSlideDescriber()
    if provider in ("local", "llamacpp", "llama.cpp"):
        return LlamaCppSlideDescriber()
    raise ValueError(f"Unknown description provider: {provider!r}")
