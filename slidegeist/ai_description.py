"""AI slide description via llama.cpp server (OpenAI-compatible vision API)."""

import base64
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
        "Extract ALL content with maximum precision to enable accurate reconstruction "
        "in ANY format (PowerPoint, LaTeX Beamer, Markdown, Jupyter, Quarto, Manim). "
        "Slides may contain BOTH handwritten AND machine-printed content. "
        "Diagrams may need recreation as SVG, TikZ, Manim, or other vector formats. "
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
        context_parts.append(f"Speaker transcript: {transcript[:500]}")

    if ocr_text:
        context_parts.append(
            f"OCR text (may contain artifacts): {ocr_text[:500]}"
        )

    context = "\n".join(context_parts) if context_parts else "No context available"

    return f"""Describe this slide so another AI can recreate it exactly. This slide may contain HANDWRITTEN text, formulas, and figures in addition to printed text.

Reference context (may contain OCR artifacts):
{context}

Output exactly 5 numbered sections:

1. TITLE
[If visible: exact title text. If not visible: infer descriptive title from content (2-8 words)]

2. TEXT CONTENT
[List ALL visible text verbatim in order (top to bottom, left to right)]
[Specify if handwritten or printed for each text block]
[Include: headings, body text, bullet points, labels, annotations]
[Format: Use markdown bullets for lists, preserve line breaks]

3. FORMULAS
[Every mathematical equation in LaTeX notation]
[Specify if handwritten or printed]
[Format: One equation per line with $...$ for inline or $$...$$ for display]
[Include: variable definitions, units, equation numbers if present]
[If no formulas: write "None"]

4. VISUAL ELEMENTS
[Describe every diagram, plot, graph, or illustration for recreation]
[Specify: type (flowchart/plot/diagram), spatial layout (top-left/center/etc), components (boxes/arrows/curves), colors, labels]
[Note if hand-drawn or computer-generated]
[If no visual elements: write "None"]

5. LAYOUT
[Overall structure: single-column/two-column/grid]
[Spatial relationships: what's above/below/beside what]
[Hierarchy: title size, heading levels, emphasis]

END"""


def _image_to_data_url(image_path: Path, max_dimension: int = 1280) -> str:
    """Load image, resize if needed, and return as base64 data URL.

    Args:
        image_path: Path to image file.
        max_dimension: Max width/height in pixels.

    Returns:
        Data URL string (data:image/jpeg;base64,...).
    """
    from PIL import Image

    image = Image.open(image_path)

    if image.width > max_dimension or image.height > max_dimension:
        scale = min(max_dimension / image.width, max_dimension / image.height)
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        logger.debug(f"Resized {image_path.name} to {new_size}")

    import io

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


class LlamaCppDescriber:
    """AI slide describer via llama.cpp OpenAI-compatible vision API."""

    def __init__(
        self,
        llm_url: str = DEFAULT_LLM_URL,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> None:
        self.name = "llama.cpp (Qwen3.5-VL)"
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

        system_instruction = get_system_instruction()
        user_text = get_user_prompt(transcript, ocr_text)
        data_url = _image_to_data_url(image_path)

        messages = [
            {"role": "system", "content": system_instruction},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                    {"type": "text", "text": user_text},
                ],
            },
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
