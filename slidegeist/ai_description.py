"""AI slide description via llama.cpp server (OpenAI-compatible API)."""

import json
import logging
import re
import time
from pathlib import Path
from urllib.request import Request, urlopen

from slidegeist.constants import DEFAULT_LLM_URL

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Remove artifacts and normalize spacing."""
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
    return (
        "You are an expert at analyzing academic and scientific presentation slides. "
        "Based on the OCR text and speaker transcript, produce a structured description "
        "that enables accurate reconstruction of the slide in ANY format "
        "(PowerPoint, LaTeX Beamer, Markdown, Jupyter, Quarto, Manim). "
        "Your structured output will be parsed programmatically by downstream tools. "
        "Prioritize completeness and accuracy over brevity."
    )


def get_user_prompt(transcript: str, ocr_text: str) -> str:
    """Build comprehensive prompt for slide description.

    Args:
        transcript: Speaker transcript (may be empty)
        ocr_text: Tesseract OCR output (may contain artifacts)
    """
    context_parts = []

    if transcript:
        context_parts.append(f"Speaker transcript: {transcript[:1000]}")

    if ocr_text:
        context_parts.append(
            f"OCR text from slide (may contain artifacts): {ocr_text[:1000]}"
        )

    if not context_parts:
        return "No context available for this slide."

    context = "\n".join(context_parts)

    return f"""Based on the following context, describe this lecture slide so another AI can recreate it.

{context}

Output exactly 5 numbered sections:

1. TITLE
[Infer the slide title from context (2-8 words)]

2. TEXT CONTENT
[Reconstruct the slide text from OCR and transcript]
[Format: Use markdown bullets for lists, preserve structure]

3. FORMULAS
[Any mathematical equations mentioned, in LaTeX notation]
[If no formulas: write "None"]

4. VISUAL ELEMENTS
[Infer diagrams, plots, or illustrations from transcript context]
[If no visual elements: write "None"]

5. LAYOUT
[Infer overall structure from context]

END"""


class LlamaCppDescriber:
    """AI slide describer via llama.cpp OpenAI-compatible API.

    Uses OCR text and transcript as input context (text-only model).
    """

    def __init__(
        self,
        llm_url: str = DEFAULT_LLM_URL,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> None:
        self.name = "llama.cpp (Qwen3.5-9B)"
        self.llm_url = llm_url
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._available: bool | None = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available

        try:
            req = Request(f"{self.llm_url}/health", method="GET")
            with urlopen(req, timeout=2) as resp:
                self._available = resp.status == 200
        except Exception:
            self._available = False

        if self._available:
            logger.info(f"llama.cpp server available at {self.llm_url}")
        else:
            logger.warning(f"llama.cpp server not available at {self.llm_url}")
        return self._available

    def describe(self, image_path: Path, transcript: str, ocr_text: str = "") -> str:
        if not self.is_available():
            return ""

        if not transcript and not ocr_text:
            logger.debug(f"No context for {image_path.name}, skipping AI description")
            return ""

        system_instruction = get_system_instruction()
        user_text = get_user_prompt(transcript, ocr_text)

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_text},
        ]

        payload = json.dumps({
            "model": "qwen3.5-9b",
            "messages": messages,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": 0.8,
            "stream": False,
        }).encode("utf-8")

        url = f"{self.llm_url}/v1/chat/completions"
        req = Request(url, data=payload, method="POST")
        req.add_header("Content-Type", "application/json")

        logger.info(f"Generating description (max {self.max_new_tokens} tokens)...")
        start_time = time.time()

        with urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        elapsed = time.time() - start_time
        usage = result.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        tokens_per_sec = completion_tokens / elapsed if elapsed > 0 else 0
        logger.info(
            f"Generation complete in {elapsed:.1f}s "
            f"({completion_tokens} tokens, {tokens_per_sec:.1f} tok/s)"
        )

        content = result["choices"][0]["message"]["content"]
        return clean_text(content)


def build_ai_describer(llm_url: str = DEFAULT_LLM_URL) -> LlamaCppDescriber | None:
    """Build AI describer using llama.cpp server.

    Args:
        llm_url: Base URL of the llama.cpp server.

    Returns:
        LlamaCppDescriber if available, None otherwise.
    """
    import os

    if os.getenv("SLIDEGEIST_DISABLE_QWEN", "").lower() in {"1", "true", "yes"}:
        return None

    describer = LlamaCppDescriber(llm_url=llm_url)
    if describer.is_available():
        return describer

    return None
