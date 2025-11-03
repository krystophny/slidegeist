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


class TorchQwen3Describer:
    """AI slide describer using Qwen3-VL-8B via PyTorch with full CUDA support."""

    MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

    def __init__(
        self,
        max_new_tokens: int = 2048,
        temperature: float = 0.3,
        device: str = "auto",
    ) -> None:
        self.name = "Qwen3-VL-8B (PyTorch)"
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device
        self._model = None
        self._processor = None
        self._available = False

        try:
            import torch  # type: ignore[import-untyped]
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig  # type: ignore[import-untyped]

            self._torch = torch
            self._Qwen3VLForConditionalGeneration = Qwen3VLForConditionalGeneration
            self._AutoProcessor = AutoProcessor
            self._BitsAndBytesConfig = BitsAndBytesConfig
            self._available = True
            logger.info("PyTorch with transformers available for Qwen3-VL descriptions")
        except ImportError as e:
            logger.debug(f"PyTorch/transformers not installed: {e}")

    def is_available(self) -> bool:
        return self._available

    def describe(self, image_path: Path, transcript: str, ocr_text: str = "") -> str:
        if not self._available:
            return ""

        self._ensure_loaded()
        if self._model is None or self._processor is None:
            return ""

        system_instruction = get_system_instruction()
        user_text = get_user_prompt(transcript, ocr_text)

        # Load image
        from PIL import Image  # type: ignore[import-untyped]
        image = Image.open(image_path)

        # Build messages
        messages = [
            {"role": "system", "content": system_instruction},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_text},
                ],
            },
        ]

        # Process inputs
        text_prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text_prompt],
            images=[image],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Generate
        with self._torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True if self.temperature > 0 else False,
            )

        # Decode
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs["input_ids"], output_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return clean_text(output_text)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        logger.info(f"Loading {self.MODEL_ID}...")

        # Determine device
        if self.device == "auto":
            if self._torch.cuda.is_available():
                self._device = "cuda"
                logger.info(f"Using CUDA GPU: {self._torch.cuda.get_device_name(0)}")
            else:
                self._device = "cpu"
                logger.info("Using CPU (no CUDA available)")
        else:
            self._device = self.device

        # Load model and processor with 8-bit quantization
        import gc
        self._torch.cuda.empty_cache()
        gc.collect()

        if self._device == "cuda":
            # Use 8-bit quantization for CUDA (fits comfortably in 16GB VRAM)
            bnb_config = self._BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=self._torch.float16,
            )
            self._model = self._Qwen3VLForConditionalGeneration.from_pretrained(
                self.MODEL_ID,
                quantization_config=bnb_config,
                device_map="auto",
            )
        else:
            # CPU: load in full precision
            self._model = self._Qwen3VLForConditionalGeneration.from_pretrained(
                self.MODEL_ID,
                dtype=self._torch.float32,
                device_map="cpu",
            )

        self._processor = self._AutoProcessor.from_pretrained(self.MODEL_ID)

        logger.info(f"Model loaded on {self._device}")


def build_ai_describer() -> TorchQwen3Describer | None:
    """Build AI describer using PyTorch."""
    import os

    if os.getenv("SLIDEGEIST_DISABLE_QWEN", "").lower() in {"1", "true", "yes"}:
        return None

    describer = TorchQwen3Describer()
    if describer.is_available():
        return describer

    return None
