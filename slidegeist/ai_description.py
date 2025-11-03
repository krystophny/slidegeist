"""AI slide description for reconstruction."""

import logging
import re
from pathlib import Path

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

    return f"""Analyze this presentation slide and extract ALL content in a structured format.

IMPORTANT: This slide may contain BOTH handwritten and machine-printed text.

Generate a complete description with these sections:

1. TOPIC: Main subject (1-3 words, e.g., "Plasma Confinement", "Maxwell Equations")

2. TITLE: Suggested slide title (extract from slide or infer from content)

3. TEXT_CONTENT:
   - ALL machine-printed text (typed, laser-printed)
   - ALL handwritten text (pen, marker, chalk)
   - Preserve EXACT wording, line breaks, and formatting
   - Note which parts are handwritten vs. printed
   - Include annotations, labels, and captions

4. FORMULAS:
   - ALL mathematical/physics expressions in LaTeX notation
   - Include inline formulas (e.g., \\( E = mc^2 \\)) and display equations
   - Preserve equation numbering if present
   - Include units, subscripts, superscripts
   - Handle Greek letters, vectors, tensors, operators (∇, ∂, ∫, etc.)

5. VISUAL_ELEMENTS:
   - Diagrams: describe topology, connections, flow
   - Plots/graphs: axes, labels, curves, data points
   - Flowcharts: boxes, decision points, arrows
   - Scientific diagrams: field lines, vectors, coordinate systems
   - Describe structure precisely for recreation as SVG, TikZ, Manim, or other formats
   - Include ALL arrows, boxes, circles, annotations with relative positions
   - Specify colors, line styles, markers if visible

6. TABLES:
   - Full structure: number of rows, columns, headers
   - ALL cell contents exactly as shown
   - Alignment, borders, merged cells
   - Units in headers if present

7. SYMBOLS:
   - Special symbols, icons, markers not covered above
   - Greek letters in text context
   - Mathematical operators, logic symbols
   - Physical constants notation

8. LAYOUT:
   - Spatial arrangement (top-to-bottom, left-to-right, multi-column)
   - Alignment, spacing, indentation
   - Visual hierarchy (title, sections, bullet points)
   - Relative positioning of elements

Context (reference only, may contain OCR errors):
{context}

OUTPUT REQUIREMENTS:
- Be EXHAUSTIVE - capture every visible element
- Use standard notation (LaTeX for math, structured descriptions for visuals)
- Format for downstream parsing by automated tools
- Prioritize accuracy over brevity"""


class LlamaCppQwen3Describer:
    """AI slide describer using Qwen3-VL via llama.cpp (CUDA/CPU with GPU offload)."""

    # Qwen3-VL-30B-A3B GGUF models from unsloth
    MODEL_ID = "unsloth/Qwen3-VL-30B-A3B-Instruct-GGUF"
    MODEL_FILE = "Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf"
    MMPROJ_FILE = "mmproj-Qwen3-VL-30B-A3B-Instruct-f16.gguf"

    def __init__(
        self,
        max_new_tokens: int = 2048,
        temperature: float = 0.3,
        n_gpu_layers: int = 50,
        main_gpu: int = 0,
    ) -> None:
        self.name = "Qwen3-VL-30B (llama.cpp)"
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.n_gpu_layers = n_gpu_layers
        self.main_gpu = main_gpu
        self._llm = None
        self._available = False

        try:
            import llama_cpp  # type: ignore[import-untyped]

            self._llama_cpp = llama_cpp
            self._available = True
            logger.info("llama-cpp-python available for Qwen3-VL descriptions")
        except ImportError:
            logger.debug("llama-cpp-python not installed")

    def is_available(self) -> bool:
        return self._available

    def describe(self, image_path: Path, transcript: str, ocr_text: str = "") -> str:
        if not self._available:
            return ""

        self._ensure_loaded()
        if self._llm is None:
            return ""

        system_instruction = get_system_instruction()
        user_text = get_user_prompt(transcript, ocr_text)

        # Convert image to data URI for llama-cpp-python
        import base64

        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        image_uri = f"data:image/jpeg;base64,{image_data}"

        messages = [
            {"role": "system", "content": system_instruction},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_uri}},
                    {"type": "text", "text": user_text},
                ],
            },
        ]

        response = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )

        raw_output = response["choices"][0]["message"]["content"]
        return clean_text(raw_output)

    def _ensure_loaded(self) -> None:
        if self._llm is not None:
            return

        from llama_cpp.llama_chat_format import Qwen25VLChatHandler

        logger.info(f"Loading {self.MODEL_ID}...")
        logger.info(
            f"GPU layers: {self.n_gpu_layers}, "
            f"GPU: {self.main_gpu}, "
            f"Max tokens: {self.max_new_tokens}"
        )

        # Download model and mmproj if needed
        model_path = self._download_model(self.MODEL_FILE)
        mmproj_path = self._download_model(self.MMPROJ_FILE)

        # Initialize chat handler with mmproj
        chat_handler = Qwen25VLChatHandler(clip_model_path=str(mmproj_path), verbose=False)

        # Load model with GPU offload
        self._llm = self._llama_cpp.Llama(
            model_path=str(model_path),
            chat_handler=chat_handler,
            n_gpu_layers=self.n_gpu_layers,
            main_gpu=self.main_gpu,
            n_ctx=4096,
            verbose=False,
        )

    def _download_model(self, filename: str) -> Path:
        """Download model file from HuggingFace if needed."""
        from pathlib import Path

        cache_dir = Path.home() / ".cache" / "slidegeist" / "models"
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_file = cache_dir / filename

        if model_file.exists():
            logger.info(f"Using cached model: {model_file}")
            return model_file

        logger.info(f"Downloading {filename} from HuggingFace...")
        try:
            from huggingface_hub import hf_hub_download  # type: ignore[import-untyped]

            downloaded_path = hf_hub_download(
                repo_id=self.MODEL_ID, filename=filename, cache_dir=str(cache_dir)
            )
            return Path(downloaded_path)
        except ImportError:
            raise ImportError(
                "huggingface_hub not installed. Install with: pip install huggingface-hub"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to download {filename}: {e}") from e


def build_ai_describer() -> LlamaCppQwen3Describer | None:
    """Build AI describer using llama.cpp."""
    import os

    if os.getenv("SLIDEGEIST_DISABLE_QWEN", "").lower() in {"1", "true", "yes"}:
        return None

    describer = LlamaCppQwen3Describer()
    if describer.is_available():
        return describer

    return None
