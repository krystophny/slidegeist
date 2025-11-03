"""Audio transcription using whisper.cpp."""

import logging
import platform
import sys
import time
from pathlib import Path
from typing import TypedDict

from slidegeist.constants import (
    DEFAULT_DEVICE,
    DEFAULT_WHISPER_MODEL,
)

logger = logging.getLogger(__name__)


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
    """Transcribe video audio using whisper.cpp.

    Args:
        video_path: Path to the video file.
        model_size: Whisper model size: tiny, base, small, medium, large-v3, large-v2, large.
        device: Device to use: 'cpu', 'cuda', or 'auto' (auto-detects best available).
        compute_type: Unused, kept for API compatibility.

    Returns:
        Dictionary with language and segments containing timestamped text.

    Raises:
        ImportError: If pywhispercpp is not installed.
        Exception: If transcription fails.
    """
    try:
        from pywhispercpp.model import Model  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError("pywhispercpp not installed. Install with: pip install pywhispercpp")

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # whisper.cpp auto-detects CUDA, device parameter kept for API compatibility
    if device == "auto":
        logger.info("Device: auto (whisper.cpp will auto-detect CUDA/CPU)")

    # Map model names to pywhispercpp format
    model_map = {
        "large-v3": "large-v3",
        "large-v3-turbo": "large-v3-turbo",
        "large-v2": "large-v2",
        "large": "large-v2",
        "medium": "medium",
        "small": "small",
        "base": "base",
        "tiny": "tiny",
    }
    whisper_model = model_map.get(model_size, model_size)

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

    logger.info(f"Loading Whisper model: {whisper_model} on {device}")
    start_time = time.time()

    # Initialize model (whisper.cpp auto-detects CUDA)
    # n_threads=0 means auto-detect optimal thread count
    model = Model(
        model=whisper_model,
        n_threads=0,
    )

    logger.info(f"Transcribing: {video_path.name}")

    # Transcribe (whisper.cpp doesn't provide word-level timestamps via pywhispercpp)
    segments = model.transcribe(
        media=str(video_path),
        new_segment_callback=None,  # Could add progress callback here
    )

    # Convert pywhispercpp output to our format
    segments_list: list[Segment] = []
    detected_language = "unknown"

    for segment in segments:
        # pywhispercpp doesn't provide word-level timestamps
        segments_list.append({
            "start": segment.t0 / 1000.0,  # Convert ms to seconds
            "end": segment.t1 / 1000.0,
            "text": segment.text.strip(),
            "words": [],  # No word-level data available
        })

    # Clear progress line and show final stats
    try:
        import shutil
        terminal_width = shutil.get_terminal_size((80, 20)).columns
    except Exception:
        terminal_width = 120

    print("\r" + " " * terminal_width + "\r", end="", file=sys.stderr, flush=True)

    total_time = time.time() - start_time
    speed_factor = video_duration / total_time if video_duration and total_time > 0 else 0

    logger.info(
        f"Transcription complete: {len(segments_list)} segments, "
        f"time: {total_time / 60:.1f}min"
    )
    if speed_factor > 0:
        logger.info(f"Average speed: {speed_factor:.2f}x realtime")

    return {"language": detected_language, "segments": segments_list}
