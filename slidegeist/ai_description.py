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


class TorchQwen3Describer:
    """AI slide describer using Qwen3-VL-8B via PyTorch with 4-bit quantization."""

    MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

    def __init__(
        self,
        max_new_tokens: int = 2048,  # Enough for complete descriptions with formulas/diagrams
        temperature: float = 0.7,  # Qwen3-VL recommended for vision tasks
        top_p: float = 0.8,  # Qwen3-VL recommended
        top_k: int = 20,  # Qwen3-VL recommended
        device: str = "auto",
    ) -> None:
        self.name = "Qwen3-VL-8B (PyTorch 8-bit)"
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
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

        # Load image (Qwen3-VL accepts PIL Image objects directly)
        from PIL import Image  # type: ignore[import-untyped]
        image = Image.open(image_path)

        # Resize to reduce vision tokens (1 token per 28x28 or 32x32 pixels)
        # Target: ~1200 tokens max (1024x768 = ~1200 tokens vs 1920x1080 = ~2500 tokens)
        max_dimension = 1280  # Good balance: readable formulas, faster processing
        if image.width > max_dimension or image.height > max_dimension:
            # Preserve aspect ratio
            scale = min(max_dimension / image.width, max_dimension / image.height)
            new_size = (int(image.width * scale), int(image.height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            logger.debug(f"Resized image from {image_path.name} to {new_size} for faster processing")

        # Build messages (format: system message as string, user content as list)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_instruction}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_text},
                ],
            },
        ]

        # Process inputs using official API (from QwenLM/Qwen3-VL repo)
        logger.debug("Processing inputs...")
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        logger.debug(f"inputs type: {type(inputs)}, keys: {inputs.keys() if isinstance(inputs, dict) else 'not a dict'}")
        inputs = inputs.to(self._model.device)

        # Generate with Qwen3-VL recommended parameters
        logger.info(f"Generating description (max {self.max_new_tokens} tokens, temp={self.temperature})...")
        import time
        import sys
        start_time = time.time()

        # Custom streamer that prints tokens without newlines for real-time feedback
        streamer = None
        if logger.isEnabledFor(logging.DEBUG):
            from transformers import TextStreamer  # type: ignore[import-untyped]
            # TextStreamer prints to sys.stdout by default, let it print directly
            streamer = TextStreamer(
                self._processor.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )
            sys.stdout.write("\nGenerating: ")
            sys.stdout.flush()

        with self._torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                do_sample=True,
                repetition_penalty=1.05,  # Slightly higher to reduce repetition
                streamer=streamer,
            )

        if streamer is not None:
            sys.stdout.write("\n")
            sys.stdout.flush()

        elapsed = time.time() - start_time
        tokens_generated = len(output_ids[0]) - len(inputs["input_ids"][0])
        tokens_per_sec = tokens_generated / elapsed if elapsed > 0 else 0
        logger.info(f"Generation complete in {elapsed:.1f}s ({tokens_generated} tokens, {tokens_per_sec:.1f} tok/s)")

        # Decode (trim input prompt from output)
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], output_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        logger.debug(f"Generated description:\n{output_text}")

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
            # Use 8-bit quantization for CUDA (fits in 16GB VRAM: ~8-10GB)
            quantization_config = self._BitsAndBytesConfig(load_in_8bit=True)
            self._model = self._Qwen3VLForConditionalGeneration.from_pretrained(
                self.MODEL_ID,
                quantization_config=quantization_config,
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
