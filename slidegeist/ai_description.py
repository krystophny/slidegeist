"""AI slide description for reconstruction."""

import logging
import platform
from pathlib import Path

logger = logging.getLogger(__name__)


def get_system_instruction() -> str:
    return (
        "You are an expert at analyzing academic and professional presentation slides. "
        "Describe the slide in extreme detail so another AI could recreate it perfectly in LaTeX Beamer. "
        "Include: all text verbatim, layout structure, font sizes/styles, colors, "
        "mathematical formulas (in LaTeX), diagrams (describe for TikZ/SVG recreation), "
        "charts/graphs (describe data and styling), images (describe content and placement), "
        "bullet points, numbering, alignment, spacing, and any visual elements."
    )


def get_user_prompt(transcript: str) -> str:
    context_info = f"\n\nTranscript context: {transcript[:500]}" if transcript else ""
    return (
        "Analyze this presentation slide and provide an exhaustive description "
        "that would allow another AI to recreate it pixel-perfectly in LaTeX Beamer, "
        "with TikZ/SVG for diagrams and proper formatting for all visual elements."
        f"{context_info}"
    )


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

    def describe(self, image_path: Path, transcript: str) -> str:
        if not self._available:
            return ""

        self._ensure_loaded()
        if self._model is None or self._processor is None:
            return ""

        system_instruction = get_system_instruction()
        user_text = get_user_prompt(transcript)

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

    MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

    def __init__(self, max_new_tokens: int = 2048, temperature: float = 0.3) -> None:
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._model = None
        self._processor = None
        self._available = False
        self._device = "cpu"

        try:
            import torch  # type: ignore[import-untyped]
            from transformers import (  # type: ignore[import-untyped]
                AutoProcessor,
                Qwen3VLForConditionalGeneration,
            )

            self._torch = torch
            self._qwen3vl_class = Qwen3VLForConditionalGeneration
            self._autoprocessor_class = AutoProcessor

            if torch.cuda.is_available():
                self._device = "cuda"
                self.name = "Qwen3-VL-8B (CUDA)"
                logger.info("PyTorch CUDA available for Qwen3-VL")
            else:
                self._device = "cpu"
                self.name = "Qwen3-VL-8B (CPU)"
                logger.info("PyTorch CPU available for Qwen3-VL")

            self._available = True
        except ImportError:
            logger.debug("PyTorch or transformers not available")

    def is_available(self) -> bool:
        return self._available

    def describe(self, image_path: Path, transcript: str) -> str:
        if not self._available:
            return ""

        self._ensure_loaded()
        if self._model is None or self._processor is None:
            return ""

        system_instruction = get_system_instruction()
        user_text = get_user_prompt(transcript)
        combined_text = f"{system_instruction}\n\n{user_text}"

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "url": f"file://{image_path.absolute()}"},
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

        return output_text[0].strip() if output_text else ""

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        logger.info(f"Loading {self.MODEL_ID}...")

        if self._device == "cuda":
            dtype = self._torch.float16
            device_map = "auto"
        else:
            dtype = self._torch.float32
            device_map = None

        self._model = self._qwen3vl_class.from_pretrained(
            self.MODEL_ID, torch_dtype=dtype, device_map=device_map
        )

        if device_map is None:
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
