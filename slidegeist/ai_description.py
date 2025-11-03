"""AI slide description for reconstruction."""

import logging
import platform
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
        "You are an expert at analyzing presentation slides for AI-to-AI processing. "
        "Provide structured, machine-readable descriptions that enable reconstruction "
        "in multiple formats: LaTeX Beamer, SVG, JavaScript, Jupyter notebooks, Quarto markdown. "
        "Focus on precision and completeness over brevity."
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

    return f"""Analyze this slide and generate a structured description with these sections:

1. TOPIC: Main subject (1-3 words)
2. TITLE: Suggested slide title
3. TEXT_CONTENT:
   - All machine-written text (typed, printed)
   - All handwritten text
   - Preserve exact wording and formatting
4. FORMULAS:
   - Mathematical expressions in LaTeX notation
   - Include inline and display equations
5. VISUAL_ELEMENTS:
   - Diagrams, flowcharts, plots, charts
   - Describe structure for SVG/TikZ recreation
   - Arrows, boxes, annotations with positions
6. TABLES:
   - Structure (rows, columns, headers)
   - All cell contents
7. SYMBOLS:
   - Special symbols, icons, markers
8. LAYOUT:
   - Spatial arrangement
   - Alignment, spacing, hierarchy

Context:
{context}

Provide complete, precise descriptions suitable for programmatic processing."""


class MlxQwen3Describer:
    """AI slide describer using Qwen3-VL via MLX (Apple Silicon only)."""

    MODEL_ID = "lmstudio-community/Qwen3-VL-8B-Instruct-MLX-4bit"

    def __init__(self, max_new_tokens: int = 2048, temperature: float = 0.3) -> None:
        self.name = "Qwen3-VL-8B (MLX)"
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._model = None
        self._processor = None
        self._config = None
        self._available = False

        if platform.system() != "Darwin" or platform.machine() != "arm64":
            return

        try:
            import mlx_vlm  # type: ignore
            from mlx_vlm import generate, load  # type: ignore
            from mlx_vlm.prompt_utils import apply_chat_template  # type: ignore

            self._load = load
            self._generate = generate
            self._apply_chat_template = apply_chat_template
            self.version = getattr(mlx_vlm, "__version__", None)
            self._available = True
            logger.info("MLX-VLM available for Qwen3-VL descriptions")
        except ImportError:
            logger.debug("mlx-vlm not installed")

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

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_instruction}]},
            {"role": "user", "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": user_text},
            ]},
        ]

        formatted = self._apply_chat_template(  # type: ignore
            self._processor, self._config, messages, add_generation_prompt=True
        )

        output = self._generate(  # type: ignore
            self._model,
            self._processor,
            formatted,
            images=[str(image_path)],
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            verbose=False,
        )

        raw_output = self._extract_text(output)
        return clean_text(raw_output)

    def _extract_text(self, output: str | dict) -> str:
        """Extract text from model output."""
        if isinstance(output, str):
            return output.strip()
        elif isinstance(output, dict):
            return str(output.get("choices", [{}])[0].get("text", "")).strip()
        return str(output).strip()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        logger.info(f"Loading {self.MODEL_ID}...")
        self._model, self._processor = self._load(self.MODEL_ID)  # type: ignore
        self._config = getattr(self._model, "config", None)


class TorchQwen3Describer:
    """AI slide describer using Qwen3-VL via PyTorch (CUDA/CPU)."""

    MODEL_ID = "QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ"

    def __init__(self, max_new_tokens: int = 2048, temperature: float = 0.3) -> None:
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._model = None
        self._processor = None
        self._available = False
        self._device = "cpu"

        try:
            import torch  # type: ignore[import-untyped]
            from transformers import (  # type: ignore[import-untyped,import-not-found]
                AutoProcessor,
                Qwen3VLMoeForConditionalGeneration,
            )

            self._torch = torch
            self._qwen3vl_class = Qwen3VLMoeForConditionalGeneration
            self._autoprocessor_class = AutoProcessor

            if torch.cuda.is_available():
                self._device = "cuda"
                self.name = "Qwen3-VL-30B-A3B-AWQ (CUDA+CPU)"
                logger.info("PyTorch CUDA available for Qwen3-VL")
            else:
                self._device = "cpu"
                self.name = "Qwen3-VL-30B-A3B-AWQ (CPU)"
                logger.info("PyTorch CPU available for Qwen3-VL")

            self._available = True
        except ImportError:
            logger.debug("PyTorch or transformers not available")

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
        combined_text = f"{system_instruction}\n\n{user_text}"

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path.absolute())},
                {"type": "text", "text": combined_text},
            ]
        }]

        inputs = self._processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        )
        inputs.pop("token_type_ids", None)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        generated_ids = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        raw_output = output_text[0].strip() if output_text else ""
        return clean_text(raw_output)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        logger.info(f"Loading {self.MODEL_ID}...")

        if self._device == "cuda":
            # AWQ quantized model with GPU + CPU offloading
            # Allocate 14GB to GPU, rest to CPU (model is ~17GB total)
            import psutil
            cpu_memory_gb = int(psutil.virtual_memory().available / (1024**3))

            max_memory = {
                0: "14GiB",  # Leave 2GB for other processes and activations
                "cpu": f"{min(cpu_memory_gb - 8, 32)}GiB"  # Reserve 8GB for system
            }

            logger.info(f"Loading with GPU+CPU offload: GPU=14GB, CPU={max_memory['cpu']}")

            self._model = self._qwen3vl_class.from_pretrained(
                self.MODEL_ID,
                torch_dtype="auto",
                device_map="auto",
                max_memory=max_memory
            )
        else:
            # CPU only
            logger.info("Loading on CPU (this will be slow)")
            self._model = self._qwen3vl_class.from_pretrained(
                self.MODEL_ID,
                torch_dtype=self._torch.float32
            )
            if self._model is not None:
                self._model = self._model.to(self._device)

        self._processor = self._autoprocessor_class.from_pretrained(self.MODEL_ID)


def build_ai_describer() -> MlxQwen3Describer | TorchQwen3Describer | None:
    """Build AI describer using best available backend."""
    import os

    if os.getenv("SLIDEGEIST_DISABLE_QWEN", "").lower() in {"1", "true", "yes"}:
        return None

    mlx = MlxQwen3Describer()
    if mlx.is_available():
        return mlx

    torch_desc = TorchQwen3Describer()
    if torch_desc.is_available():
        return torch_desc

    return None
