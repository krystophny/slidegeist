"""Audio transcription using OpenAI Whisper (PyTorch backend)."""

import logging
import platform
import time
from pathlib import Path
from typing import TypedDict

from slidegeist.constants import (
    COMPRESSION_RATIO_THRESHOLD,
    DEFAULT_DEVICE,
    DEFAULT_WHISPER_MODEL,
    LOG_PROB_THRESHOLD,
    NO_SPEECH_THRESHOLD,
)

logger = logging.getLogger(__name__)


def is_mlx_available() -> bool:
    """Check if MLX is available (Apple Silicon Mac).

    Returns:
        True if running on Apple Silicon with MLX support, False otherwise.
    """
    # Allow user to completely disable MLX if it causes hard crashes
    import os
    if os.getenv("SLIDEGEIST_DISABLE_MLX", "").lower() in {"1", "true", "yes"}:
        logger.info("MLX disabled via SLIDEGEIST_DISABLE_MLX environment variable")
        return False

    # Check if we're on macOS ARM64 (Apple Silicon)
    if platform.system() != "Darwin":
        return False
    if platform.machine() != "arm64":
        return False

    # Check if mlx-whisper is importable without actually importing it
    # NOTE: Even find_spec() can trigger hard crashes if MLX C++ bindings are corrupted
    # If experiencing hard crashes (macOS crash dialog), set SLIDEGEIST_DISABLE_MLX=1
    try:
        import importlib.util
        spec = importlib.util.find_spec("mlx_whisper")
        if spec is None:
            return False

        # Additional safety: check if we can at least import the package
        # This is still risky but necessary for validation
        logger.debug("MLX package found, attempting validation import")
        return True
    except (ImportError, ValueError, AttributeError):
        return False
    except Exception as e:
        # Catch any other errors including potential crashes during spec lookup
        logger.warning(f"MLX detection failed: {e}")
        return False


def is_cuda_available() -> bool:
    """Check if CUDA GPU is available for PyTorch.

    Returns:
        True if CUDA GPU is available and working, False otherwise.
    """
    try:
        import torch  # type: ignore[import-untyped]

        return torch.cuda.is_available()
    except (ImportError, AttributeError, RuntimeError):
        # ImportError: torch not installed
        # AttributeError: torch.cuda not available
        # RuntimeError: CUDA initialization failed
        return False


class Word(TypedDict):
    """A single word with timing information."""

    word: str
    start: float
    end: float


class Segment(TypedDict):
    """A transcript segment with timing and words."""

    start: float
    end: float
    text: str
    words: list[Word]


class TranscriptResult(TypedDict):
    """Complete transcription result."""

    language: str
    segments: list[Segment]


def transcribe_video(
    video_path: Path,
    model_size: str = DEFAULT_WHISPER_MODEL,
    device: str = DEFAULT_DEVICE,
    compute_type: str = "int8",
) -> TranscriptResult:
    """Transcribe video audio using OpenAI Whisper (PyTorch backend).

    Args:
        video_path: Path to the video file.
        model_size: Whisper model size: tiny, base, small, medium, large-v3, large-v2, large.
        device: Device to use: 'cpu', 'cuda', or 'auto' (auto-detects MLX on Apple Silicon).
        compute_type: Computation type (unused, kept for API compatibility).

    Returns:
        Dictionary with language and segments containing timestamped text.

    Raises:
        ImportError: If openai-whisper is not installed.
        Exception: If transcription fails.
    """

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Auto-detect best available device
    use_mlx = False
    if device == "auto":
        if is_mlx_available():
            use_mlx = True
            device = "cpu"  # MLX uses its own backend
            logger.info("MLX detected - using MLX-optimized Whisper for Apple Silicon")
        elif is_cuda_available():
            device = "cuda"
            logger.info("CUDA GPU detected - using GPU acceleration")
        elif platform.system() == "Darwin" and platform.machine() == "arm64":
            device = "cpu"
            logger.info(
                "Apple Silicon detected but MLX not available, using CPU. Install with: pip install mlx-whisper"
            )
        else:
            device = "cpu"
            logger.info("Auto-detected device: CPU")

    # Use MLX-optimized transcription if available
    if use_mlx:
        try:
            import mlx_whisper  # type: ignore[import-untyped, import-not-found]

            # Suppress MLX verbose debug output (only after successful import)
            try:
                logging.getLogger("mlx").setLevel(logging.WARNING)
                logging.getLogger("mlx_whisper").setLevel(logging.WARNING)
            except Exception:
                pass  # Ignore logger configuration errors

            # Map faster-whisper model names to MLX model names
            mlx_model_map = {
                "large-v3": "mlx-community/whisper-large-v3-mlx",
                "large-v2": "mlx-community/whisper-large-v2-mlx",
                "large": "mlx-community/whisper-large-v2-mlx",
                "medium": "mlx-community/whisper-medium-mlx",
                "small": "mlx-community/whisper-small-mlx",
                "base": "mlx-community/whisper-base-mlx",
                "tiny": "mlx-community/whisper-tiny-mlx",
            }
            mlx_model = mlx_model_map.get(model_size, f"mlx-community/whisper-{model_size}-mlx")

            logger.info(f"Loading MLX Whisper model: {mlx_model}")
            result = mlx_whisper.transcribe(
                str(video_path),
                path_or_hf_repo=mlx_model,
                word_timestamps=True,
            )
            # Convert MLX result to our format
            mlx_segments: list[Segment] = []
            for segment in result.get("segments", []):
                mlx_words: list[Word] = []
                for word in segment.get("words", []):
                    mlx_words.append(
                        {"word": word["word"], "start": word["start"], "end": word["end"]}
                    )
                mlx_segments.append(
                    {
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": segment["text"].strip(),
                        "words": mlx_words,
                    }
                )
            logger.info(f"MLX transcription complete: {len(mlx_segments)} segments")
            return {"language": result.get("language", "unknown"), "segments": mlx_segments}
        except ImportError as e:
            logger.warning(f"MLX import failed: {e}, falling back to openai-whisper")
            use_mlx = False
        except (KeyError, AttributeError, TypeError) as e:
            logger.warning(f"MLX data format error: {e}, falling back to openai-whisper")
            use_mlx = False
        except Exception as e:
            logger.error(f"MLX transcription crashed: {e}, falling back to openai-whisper")
            logger.debug("Full traceback:", exc_info=True)
            use_mlx = False

    # Load OpenAI Whisper model (PyTorch backend)
    import whisper  # type: ignore[import-untyped]

    logger.info(f"Loading Whisper model: {model_size} on {device}")
    model = whisper.load_model(model_size, device=device)

    # Get video duration for progress tracking
    from slidegeist.ffmpeg import get_video_duration

    try:
        video_duration = get_video_duration(video_path)
        logger.info(
            f"Video duration: {video_duration / 60:.1f} minutes ({video_duration:.1f} seconds)"
        )
    except Exception:
        video_duration = None
        logger.warning("Could not determine video duration, progress tracking will be limited")

    logger.info(f"Transcribing: {video_path.name}")
    start_time = time.time()

    # Transcribe with OpenAI Whisper with progress bar
    from tqdm import tqdm

    # OpenAI Whisper doesn't have built-in progress callbacks, so we use verbose output
    # and wrap it with tqdm to show progress based on printed segments
    import io
    import contextlib

    if video_duration and video_duration > 0:
        # Show progress bar based on estimated duration
        pbar = tqdm(total=int(video_duration), unit="s", desc="Transcribing", ncols=100)

        # Capture verbose output to update progress
        class ProgressCapture:
            def __init__(self, pbar, start_time):
                self.pbar = pbar
                self.start_time = start_time
                self.last_position = 0

            def write(self, text):
                # Parse progress from Whisper's output (shows timestamps)
                import re
                # Look for timestamp patterns like [00:01.000 --> 00:05.000]
                match = re.search(r'\[(\d+):(\d+)\.(\d+) --> (\d+):(\d+)\.(\d+)\]', text)
                if match:
                    # Calculate end time in seconds
                    end_min, end_sec = int(match.group(4)), int(match.group(5))
                    position = end_min * 60 + end_sec
                    if position > self.last_position:
                        self.pbar.update(position - self.last_position)
                        self.last_position = position

            def flush(self):
                pass

        progress_capture = ProgressCapture(pbar, start_time)

        # Redirect stderr to capture Whisper's verbose output
        import sys
        old_stderr = sys.stderr
        sys.stderr = progress_capture

        try:
            result = model.transcribe(
                str(video_path),
                word_timestamps=True,
                compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
                logprob_threshold=LOG_PROB_THRESHOLD,
                no_speech_threshold=NO_SPEECH_THRESHOLD,
                verbose=True,
            )
        finally:
            sys.stderr = old_stderr
            pbar.close()
    else:
        # No duration info, just show indeterminate progress
        result = model.transcribe(
            str(video_path),
            word_timestamps=True,
            compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
            logprob_threshold=LOG_PROB_THRESHOLD,
            no_speech_threshold=NO_SPEECH_THRESHOLD,
            verbose=False,
        )

    # Extract segments
    segments_list: list[Segment] = []
    for segment in result["segments"]:
        words_list: list[Word] = []
        if "words" in segment:
            for word in segment["words"]:
                words_list.append({"word": word["word"], "start": word["start"], "end": word["end"]})

        segments_list.append(
            {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip(),
                "words": words_list,
            }
        )

    detected_language = result.get("language", "unknown")

    # Show completion stats
    total_time = time.time() - start_time
    if video_duration and video_duration > 0:
        speed_factor = video_duration / total_time
        logger.info(
            f"Transcription complete: {len(segments_list)} segments in {total_time/60:.1f}min "
            f"({speed_factor:.2f}x realtime)"
        )
    else:
        logger.info(f"Transcription complete: {len(segments_list)} segments in {total_time/60:.1f}min")

    return {"language": detected_language, "segments": segments_list}

