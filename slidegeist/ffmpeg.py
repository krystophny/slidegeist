"""FFmpeg wrapper for video processing and scene detection."""

import logging
import shutil
import subprocess
from pathlib import Path

from slidegeist.constants import (
    DEFAULT_MIN_SCENE_LEN,
    DEFAULT_SCENE_THRESHOLD,
    DEFAULT_START_OFFSET,
)

logger = logging.getLogger(__name__)


class FFmpegError(Exception):
    """Raised when FFmpeg operations fail."""
    pass


def check_ffmpeg_available() -> bool:
    """Check if FFmpeg is installed and available in PATH.

    Returns:
        True if FFmpeg is available, False otherwise.
    """
    return shutil.which("ffmpeg") is not None


def get_video_duration(video_path: Path) -> float:
    """Get the duration of a video file in seconds.

    Args:
        video_path: Path to the video file.

    Returns:
        Duration in seconds.

    Raises:
        FFmpegError: If unable to determine video duration.
    """
    if not check_ffmpeg_available():
        raise FFmpegError("FFmpeg not found. Please install FFmpeg.")

    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        raise FFmpegError(f"Failed to get video duration: {e}")


def detect_scenes(
    video_path: Path,
    threshold: float = DEFAULT_SCENE_THRESHOLD,
    min_scene_len: float = DEFAULT_MIN_SCENE_LEN,
    start_offset: float = DEFAULT_START_OFFSET
) -> list[float]:
    """Detect slide changes using Opencast's FFmpeg-based optimization.

    Uses FFmpeg's scene filter (SAD-based) with iterative optimization to
    achieve a target number of segments based on video duration. Matches
    Opencast's video segmentation approach exactly.

    Target: 30 segments per hour (research shows 15-45 slides/hour is typical)
    Method: Iteratively adjusts threshold until segment count is within 25% of target

    Based on: Opencast VideoSegmenterServiceImpl
    (https://docs.opencast.org/r/4.x/admin/modules/videosegmentation/)

    Args:
        video_path: Path to the video file.
        threshold: Initial scene detection threshold (0-1 scale, FFmpeg scene score).
                  Lower = more sensitive. Opencast default: 0.025 (2.5%).
                  This will be automatically adjusted during optimization.
        min_scene_len: Minimum segment length in seconds (stability threshold).
                      Segments shorter than this are merged with adjacent segments.
                      Adapted from Opencast's 60s default to 2s for slide detection.
        start_offset: Skip first N seconds to avoid mouse movement during setup.

    Returns:
        List of timestamps (in seconds) where slide changes occur, sorted.

    Raises:
        FFmpegError: If video file not found or processing fails.
    """
    if not video_path.exists():
        raise FFmpegError(f"Video file not found: {video_path}")

    from slidegeist.constants import (
        DEFAULT_MAX_CYCLES,
        DEFAULT_MAX_ERROR,
        DEFAULT_SEGMENTS_PER_HOUR,
    )
    from slidegeist.ffmpeg_scene import detect_scenes_opencast

    # Get video duration to calculate target segments
    video_duration = get_video_duration(video_path)
    duration_hours = video_duration / 3600.0

    # Calculate target segments based on duration (30 segments/hour)
    target_segments = max(3, int(DEFAULT_SEGMENTS_PER_HOUR * duration_hours))

    logger.info(
        f"Video duration: {video_duration/60:.1f} min "
        f"({duration_hours:.2f} hours), targeting {target_segments} segments"
    )

    # Run Opencast-style optimization
    timestamps, final_threshold = detect_scenes_opencast(
        video_path,
        target_segments=target_segments,
        max_error=DEFAULT_MAX_ERROR,
        max_cycles=DEFAULT_MAX_CYCLES,
        initial_threshold=threshold,
        stability_threshold=min_scene_len,
        start_offset=start_offset
    )

    return timestamps


def extract_audio(
    video_path: Path,
    output_path: Path,
    sample_rate: int = 16000
) -> None:
    """Extract audio from video file as 16kHz mono WAV for whisper.cpp.

    Args:
        video_path: Path to the video file.
        output_path: Path where the audio WAV will be saved.
        sample_rate: Output sample rate in Hz (whisper.cpp expects 16kHz).

    Raises:
        FFmpegError: If audio extraction fails.
    """
    if not check_ffmpeg_available():
        raise FFmpegError("FFmpeg not found. Please install FFmpeg.")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vn",  # No video
        "-acodec", "pcm_s16le",  # 16-bit PCM
        "-ar", str(sample_rate),  # Resample to 16kHz
        "-ac", "1",  # Mono
        "-y",  # Overwrite output file
        str(output_path)
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Extracted audio to {output_path}")
    except subprocess.CalledProcessError as e:
        raise FFmpegError(f"Failed to extract audio: {e.stderr}")


def extract_frame(
    video_path: Path,
    timestamp: float,
    output_path: Path,
    image_format: str = "jpg"
) -> None:
    """Extract a single frame from a video at the specified timestamp.

    Args:
        video_path: Path to the video file.
        timestamp: Time in seconds to extract the frame.
        output_path: Path where the frame image will be saved.
        image_format: Output image format ('jpg' or 'png').

    Raises:
        FFmpegError: If frame extraction fails.
    """
    if not check_ffmpeg_available():
        raise FFmpegError("FFmpeg not found. Please install FFmpeg.")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Quality settings
    quality_args = []
    if image_format == "jpg":
        quality_args = ["-q:v", "2"]  # High quality JPEG (2-5 is good range)

    cmd = [
        "ffmpeg",
        "-ss", str(timestamp),  # Seek to timestamp
        "-i", str(video_path),
        "-frames:v", "1",  # Extract one frame
        *quality_args,
        "-strict", "unofficial",  # Allow non-standard YUV colorspace
        "-y",  # Overwrite output file
        str(output_path)
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.debug(f"Extracted frame at {timestamp}s to {output_path}")
    except subprocess.CalledProcessError as e:
        raise FFmpegError(f"Failed to extract frame: {e.stderr}")
