"""FFmpeg scene detection wrapper - Opencast-compatible implementation."""

import logging
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def detect_scenes_ffmpeg(
    video_path: Path,
    threshold: float = 0.025,
    start_offset: float = 3.0
) -> list[float]:
    """Detect scene changes using FFmpeg's scene filter.

    This uses the same scene detection method as Opencast, which computes
    a normalized SAD (Sum of Absolute Differences) score between frames.

    Args:
        video_path: Path to video file.
        threshold: Scene detection threshold (0-1 scale). Default 0.025 (2.5%).
        start_offset: Skip first N seconds.

    Returns:
        List of timestamps where scene changes occur.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # FFmpeg command to detect scenes and output metadata
    # select='gt(scene,THRESHOLD)' filters frames where scene score > threshold
    # showinfo outputs frame metadata including pts_time timestamps
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-filter:v", f"select='gt(scene,{threshold})',showinfo",
        "-f", "null",
        "-"
    ]

    logger.info(f"Running FFmpeg scene detection with threshold={threshold:.4f}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            check=True
        )

        # Parse output to extract timestamps
        # FFmpeg showinfo filter outputs: pts_time:15.16
        timestamps = []
        for line in result.stderr.splitlines():
            if 'pts_time:' in line:
                # Extract pts_time value using regex
                match = re.search(r'pts_time:([\d.]+)', line)
                if match:
                    try:
                        timestamp = float(match.group(1))
                        if timestamp >= start_offset:
                            timestamps.append(timestamp)
                    except ValueError:
                        continue

        logger.info(f"Found {len(timestamps)} scene changes")
        return sorted(timestamps)

    except subprocess.TimeoutExpired:
        raise RuntimeError("FFmpeg scene detection timed out")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed: {e.stderr}")


def merge_short_segments(
    timestamps: list[float],
    min_duration: float,
    video_duration: float
) -> list[float]:
    """Merge segments shorter than minimum duration.

    This implements Opencast's segment filtering logic which merges
    segments that are too short with adjacent segments.

    Args:
        timestamps: List of scene change timestamps.
        min_duration: Minimum segment duration (stability threshold).
        video_duration: Total video duration.

    Returns:
        Filtered list of timestamps with short segments merged.
    """
    if not timestamps:
        return []

    # Convert timestamps to segments (start, end pairs)
    segments = []
    for i in range(len(timestamps)):
        start = timestamps[i-1] if i > 0 else 0.0
        end = timestamps[i]
        segments.append((start, end))
    # Add final segment
    segments.append((timestamps[-1], video_duration))

    # Filter segments by minimum duration using greedy merging
    # Merge ALL consecutive short segments into stable segments
    merged_segments = []
    i = 0
    while i < len(segments):
        start = segments[i][0]
        merged_end = segments[i][1]

        # Greedy merge: consume consecutive segments until reaching min_duration
        j = i + 1
        while j < len(segments) and (merged_end - start) < min_duration:
            merged_end = segments[j][1]
            j += 1

        # Add the merged segment
        merged_segments.append((start, merged_end))
        i = j

    # Convert back to timestamps (cut points between segments)
    result = []
    for i in range(1, len(merged_segments)):
        result.append(merged_segments[i][0])

    return result


def detect_scenes_opencast(
    video_path: Path,
    target_segments: int = 30,
    max_error: float = 0.25,
    max_cycles: int = 3,
    initial_threshold: float = 0.025,
    stability_threshold: float = 60.0,
    start_offset: float = 3.0
) -> tuple[list[float], float]:
    """Detect scenes using Opencast's iterative optimization approach.

    This replicates Opencast's video segmentation algorithm:
    1. Run FFmpeg scene detection with initial threshold
    2. Merge segments shorter than stability threshold
    3. Check if result is within acceptable error of target
    4. If not, adjust threshold and repeat

    Args:
        video_path: Path to video file.
        target_segments: Target number of segments.
        max_error: Acceptable deviation from target (0.25 = 25%).
        max_cycles: Maximum optimization iterations.
        initial_threshold: Starting threshold (Opencast default: 0.025).
        stability_threshold: Minimum segment duration in seconds (default: 60).
        start_offset: Skip first N seconds.

    Returns:
        Tuple of (timestamps, final_threshold).
    """
    from slidegeist.ffmpeg import get_video_duration

    video_duration = get_video_duration(video_path)
    threshold = initial_threshold

    logger.info(
        f"Starting Opencast-style optimization: target={target_segments}, "
        f"initial_threshold={threshold:.4f}, stability={stability_threshold}s"
    )

    for cycle in range(max_cycles):
        # Run FFmpeg scene detection
        timestamps = detect_scenes_ffmpeg(video_path, threshold, start_offset)

        # Merge short segments
        timestamps = merge_short_segments(
            timestamps,
            stability_threshold,
            video_duration
        )

        # Count segments: N timestamps (cut points) = N+1 segments
        # (segment0-t0, t0-t1, t1-t2, ..., tN-end)
        detected_segments = len(timestamps) + 1 if timestamps else 1
        error = abs(detected_segments - target_segments) / target_segments if target_segments > 0 else 0

        logger.info(
            f"Cycle {cycle + 1}/{max_cycles}: threshold={threshold:.4f}, "
            f"segments={detected_segments} (cuts={len(timestamps)}), target={target_segments}, error={error:.2%}"
        )

        # Check if within acceptable error
        if error <= max_error:
            logger.info(f"Optimization converged in {cycle + 1} cycles")
            break

        # Adjust threshold for next iteration
        if detected_segments > target_segments:
            # Too many segments, increase threshold
            threshold *= 1.5
        else:
            # Too few segments, decrease threshold
            threshold *= 0.7

        # Clamp to reasonable range (Opencast doesn't specify, using sensible limits)
        threshold = max(0.001, min(threshold, 0.5))

    final_segments = len(timestamps) + 1 if timestamps else 1
    logger.info(
        f"Opencast optimization complete: {final_segments} segments ({len(timestamps)} cuts) with threshold {threshold:.4f}"
    )
    return timestamps, threshold
