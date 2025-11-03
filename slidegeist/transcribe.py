"""Audio transcription using faster-whisper."""

import logging
import platform
import sys
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
    # Check if we're on macOS ARM64 (Apple Silicon)
    if platform.system() != "Darwin":
        return False
    if platform.machine() != "arm64":
        return False

    # Check if mlx-whisper is importable without actually importing it
    # This avoids potential crashes during detection phase
    try:
        import importlib.util
        spec = importlib.util.find_spec("mlx_whisper")
        return spec is not None
    except (ImportError, ValueError, AttributeError):
        return False


def is_cuda_available() -> bool:
    """Check if CUDA GPU is available.

    Returns:
        True if CUDA GPU is available and working, False otherwise.
    """
    try:
        import torch  # type: ignore[import-untyped, import-not-found]  # noqa: F401

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
    """Transcribe video audio using faster-whisper.

    Args:
        video_path: Path to the video file.
        model_size: Whisper model size: tiny, base, small, medium, large-v3, large-v2, large.
        device: Device to use: 'cpu', 'cuda', or 'auto' (auto-detects MLX on Apple Silicon).
        compute_type: Computation type for CTranslate2.
                     Use 'int8' for CPU, 'float16' for GPU.

    Returns:
        Dictionary with language and segments containing timestamped text.

    Raises:
        ImportError: If faster-whisper is not installed.
        Exception: If transcription fails.
    """
    try:
        from faster_whisper import WhisperModel  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError("faster-whisper not installed. Install with: pip install faster-whisper")

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
            logger.warning(f"MLX import failed: {e}, falling back to faster-whisper")
            use_mlx = False
        except (KeyError, AttributeError, TypeError) as e:
            logger.warning(f"MLX data format error: {e}, falling back to faster-whisper")
            use_mlx = False
        except Exception as e:
            logger.error(f"MLX transcription crashed: {e}, falling back to faster-whisper")
            logger.debug("Full traceback:", exc_info=True)
            use_mlx = False

    # Adjust compute type based on device
    if device == "cuda" and compute_type == "int8":
        compute_type = "float16"

    # Use all available CPU cores (0 = auto-detect optimal number)
    # This overrides the default of 4 threads
    cpu_threads = 0
    num_workers = 1

    logger.info(f"Loading Whisper model: {model_size} on {device} (compute_type: {compute_type})")
    if device == "cpu":
        logger.info(f"CPU threads: auto-detect (all cores), num_workers: {num_workers}")

    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
        num_workers=num_workers,
    )

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

    segments_iterator, info = model.transcribe(
        str(video_path),
        word_timestamps=True,
        vad_filter=True,  # Voice activity detection for better accuracy
        compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
        log_prob_threshold=LOG_PROB_THRESHOLD,
        no_speech_threshold=NO_SPEECH_THRESHOLD,
    )

    # Convert iterator to list and extract data with progress tracking
    segments_list: list[Segment] = []
    last_progress_time = start_time
    progress_interval = 5.0  # Update progress every 5 seconds

    for segment in segments_iterator:
        words_list: list[Word] = []
        if segment.words:
            for word in segment.words:
                words_list.append({"word": word.word, "start": word.start, "end": word.end})

        segments_list.append(
            {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
                "words": words_list,
            }
        )

        # Show progress update
        current_time = time.time()
        if current_time - last_progress_time >= progress_interval:
            elapsed = current_time - start_time
            current_position = segment.end

            if video_duration and video_duration > 0:
                progress_pct = (current_position / video_duration) * 100
                speed_factor = current_position / elapsed if elapsed > 0 else 0

                # Estimate remaining time
                if speed_factor > 0:
                    remaining_duration = video_duration - current_position
                    eta_seconds = remaining_duration / speed_factor
                    eta_str = f"ETA: {eta_seconds / 60:.1f}min"
                else:
                    eta_str = "ETA: calculating..."

                # Print progress bar to console
                bar_width = 40
                filled = int(bar_width * progress_pct / 100)
                bar = "█" * filled + "░" * (bar_width - filled)

                print(
                    f"\r[{bar}] {progress_pct:.1f}% | "
                    f"Position: {current_position / 60:.1f}min/{video_duration / 60:.1f}min | "
                    f"Speed: {speed_factor:.2f}x | {eta_str}",
                    end="",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                # No duration info, just show position and speed
                speed_factor = current_position / elapsed if elapsed > 0 else 0
                print(
                    f"\rProcessed: {current_position / 60:.1f}min | "
                    f"Speed: {speed_factor:.2f}x | "
                    f"Elapsed: {elapsed / 60:.1f}min",
                    end="",
                    file=sys.stderr,
                    flush=True,
                )

            last_progress_time = current_time

    # Clear progress line and show final stats
    if video_duration:
        print("\r" + " " * 120 + "\r", end="", file=sys.stderr, flush=True)

    total_time = time.time() - start_time
    speed_factor = video_duration / total_time if video_duration and total_time > 0 else 0

    logger.info(
        f"Transcription complete: {len(segments_list)} segments, "
        f"language: {info.language}, "
        f"time: {total_time / 60:.1f}min"
    )
    if speed_factor > 0:
        logger.info(f"Average speed: {speed_factor:.2f}x realtime")

    return {"language": info.language, "segments": segments_list}
