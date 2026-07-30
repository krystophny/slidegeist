"""Global Pixel Difference method for slide detection in lecture videos.

Based on research: "An experimental comparative study on slide change detection
in lecture videos" (Eruvaram et al., 2018).

This method was shown to have the best overall performance (high recall and
precision) for both slide-only and slide+presenter lecture videos.
"""

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _preprocess_video_if_needed(
    video_path: Path, max_resolution: int, target_fps: float
) -> tuple[Path, Any | None, float]:
    """Return optimized video for processing if scaling or decimation is needed."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fps <= 0:
        fps = 30.0

    needs_processing = height > max_resolution or fps > target_fps
    if not needs_processing:
        return video_path, None, fps

    scale = max_resolution / height if height > max_resolution else 1.0
    new_width = max(2, (int(width * scale) // 2) * 2)
    new_height = max(2, (int(height * scale) // 2) * 2)

    filters = []
    if scale < 1.0:
        filters.append(f"scale={new_width}:{new_height}")

    working_fps = fps
    if fps > target_fps:
        fps_ratio = max(1, int(round(fps / target_fps)))
        working_fps = fps / fps_ratio
        filters.append(f"fps=fps={working_fps}")

    filter_str = ",".join(filters)

    logger.info(
        "Preprocessing video: %dx%d@%.2ffps -> %dx%d@%.2ffps",
        width,
        height,
        fps,
        new_width,
        new_height,
        working_fps,
    )

    temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_file.close()

    cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-vf",
        filter_str,
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "28",
        "-y",
        temp_file.name,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("FFmpeg error during preprocessing: %s", result.stderr)
        try:
            os.unlink(temp_file.name)
        except OSError:
            pass
        raise RuntimeError("Video preprocessing failed")

    return Path(temp_file.name), temp_file, working_fps


def detect_slides_pixel_diff(
    video_path: Path,
    start_offset: float = 3.0,
    min_scene_len: float = 2.0,
    threshold: float = 0.02,
    sample_interval: float = 1.0,
    max_resolution: int = 360,
    target_fps: float = 5.0,
) -> list[float]:
    """Detect slide changes using Global Pixel Difference method.

    This method binarizes frames and computes pixel-level differences,
    normalized by image size. Research shows this is the most effective
    method for lecture video slide detection.

    For speed optimization, this function pre-downscales videos and reduces FPS.
    Since we're doing binary pixel difference, quality loss is minimal.

    Args:
        video_path: Path to the video file.
        start_offset: Skip first N seconds to avoid setup mouse movement.
        min_scene_len: Minimum scene length in seconds (filters rapid changes).
        threshold: Detection threshold (0-1). Default 0.02.
                  Lower = more sensitive. Typical range: 0.015-0.05.
        sample_interval: Time interval between frames to compare (seconds).
                        Default 1.0s balances accuracy and speed.
        max_resolution: Maximum resolution (height) for processing. Videos larger
                       than this will be downscaled for faster processing.
                       Default: 360p (good balance of speed/accuracy).
        target_fps: Target FPS for processing. Lower = faster.
                   Default: 5 FPS (good for slide detection).

    Returns:
        List of timestamps (seconds) where slide changes occur.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Check video properties
    cap = cv2.VideoCapture(str(video_path))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fps <= 0:
        logger.warning("Input video reported non-positive FPS (%.3f). Falling back to 30 FPS.", fps)
        fps = 30.0

    logger.info(
        f"Detecting slides with Global Pixel Difference: start_offset={start_offset}s, "
        f"min_scene_len={min_scene_len}s, threshold={threshold}, "
        f"sample_interval={sample_interval}s, video={width}x{height}@{fps:.2f}fps"
    )

    # Pre-process video for speed if needed
    working_video = video_path
    temp_file = None
    needs_processing = height > max_resolution or fps > target_fps
    working_fps = fps

    if needs_processing:
        scale = max_resolution / height if height > max_resolution else 1.0
        new_width = int(width * scale)
        new_height = int(height * scale)
        # Make dimensions divisible by 2 for h264
        new_width = (new_width // 2) * 2
        new_height = (new_height // 2) * 2

        # Build filter string
        filters = []
        if scale < 1.0:
            filters.append(f"scale={new_width}:{new_height}")
        if fps > target_fps:
            # Use integer ratio for fps to avoid encoding issues
            fps_ratio = int(round(fps / target_fps))
            actual_fps = fps / fps_ratio
            filters.append(f"fps=fps={actual_fps}")
            working_fps = actual_fps

        filter_str = ",".join(filters)

        logger.info(
            f"Preprocessing video: {width}x{height}@{fps:.2f}fps -> "
            f"{new_width}x{new_height}@{working_fps:.2f}fps"
        )

        # Create temporary optimized video
        temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        temp_file.close()

        cmd = [
            "ffmpeg",
            "-i",
            str(video_path),
            "-vf",
            filter_str,
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "28",
            "-y",
            temp_file.name,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            raise RuntimeError("Video preprocessing failed")

        working_video = Path(temp_file.name)
        logger.info(f"Preprocessed video created at {working_video}")

    # Now process the working video
    cap = cv2.VideoCapture(str(working_video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {working_video}")

    working_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    working_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / working_fps

    logger.info(
        f"Processing: {working_width}x{working_height}@{working_fps:.2f}fps, "
        f"{total_frames} frames, {duration:.1f}s duration"
    )

    # Calculate frame sampling parameters using working_fps
    start_frame = int(start_offset * working_fps)
    frame_interval = max(1, int(round(sample_interval * working_fps)))
    min_frames_between = max(1, int(round(min_scene_len * working_fps)))
    image_size = working_width * working_height

    timestamps = []
    prev_frame_binary = None
    last_change_frame = start_frame
    frame_num = start_frame

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        while frame_num < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Process only at sample intervals
            if (frame_num - start_frame) % frame_interval != 0:
                frame_num += 1
                continue

            # Convert to grayscale and binarize
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            if prev_frame_binary is not None:
                # Compute pixel-level difference
                diff = np.abs(binary.astype(np.int16) - prev_frame_binary.astype(np.int16))
                non_zero_count = np.count_nonzero(diff)

                # Normalize by image size
                normalized_diff = non_zero_count / image_size

                # Check if difference exceeds threshold
                if normalized_diff >= threshold:
                    # Check minimum scene length constraint
                    if frame_num - last_change_frame >= min_frames_between:
                        timestamp = frame_num / working_fps
                        timestamps.append(timestamp)
                        last_change_frame = frame_num

                        logger.debug(
                            f"Slide change at {timestamp:.2f}s "
                            f"(frame {frame_num}, diff={normalized_diff:.4f})"
                        )

            prev_frame_binary = binary
            frame_num += 1

    finally:
        cap.release()

        # Clean up temp file if created
        if temp_file is not None:
            try:
                os.unlink(temp_file.name)
                logger.debug(f"Cleaned up temporary file {temp_file.name}")
            except Exception:
                pass

    logger.info(f"Found {len(timestamps)} slide changes")
    return timestamps


def detect_slides_adaptive(
    video_path: Path,
    start_offset: float = 3.0,
    min_scene_len: float = 2.0,
    threshold_range: tuple[float, float] = (0.01, 0.10),
    threshold_step: float = 0.001,
    sample_interval: float = 1.0,
    max_resolution: int = 360,
    target_fps: float = 5.0,
) -> list[float]:
    """Detect slides using adaptive threshold selection.

    Computes pixel differences once, then sweeps over threshold range to find
    optimal threshold in flat region of detection curve (stable slide count).

    Args:
        video_path: Path to video file.
        start_offset: Skip first N seconds.
        min_scene_len: Minimum scene length in seconds.
        threshold_range: (min, max) threshold values to test.
        threshold_step: Step size for threshold sweep (default 0.001).
        sample_interval: Time between compared frames.
        max_resolution: Max height for processing.
        target_fps: Target FPS for processing.

    Returns:
        List of slide change timestamps using optimal threshold.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    logger.info(
        f"Adaptive slide detection: sweeping thresholds {threshold_range[0]:.3f}-{threshold_range[1]:.3f}"
    )

    # Step 1: Preprocess video and compute ALL pixel differences once
    # (same preprocessing logic as detect_slides_pixel_diff)
    cap = cv2.VideoCapture(str(video_path))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fps <= 0:
        fps = 30.0

    # Preprocess if needed
    working_video = video_path
    temp_file = None
    needs_processing = height > max_resolution or fps > target_fps
    working_fps = fps

    if needs_processing:
        scale = max_resolution / height if height > max_resolution else 1.0
        new_width = int(width * scale)
        new_height = int(height * scale)
        new_width = (new_width // 2) * 2
        new_height = (new_height // 2) * 2

        filters = []
        if scale < 1.0:
            filters.append(f"scale={new_width}:{new_height}")
        if fps > target_fps:
            fps_ratio = int(round(fps / target_fps))
            actual_fps = fps / fps_ratio
            filters.append(f"fps=fps={actual_fps}")
            working_fps = actual_fps

        filter_str = ",".join(filters)

        temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        temp_file.close()

        cmd = [
            "ffmpeg",
            "-i",
            str(video_path),
            "-vf",
            filter_str,
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "28",
            "-y",
            temp_file.name,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError("Video preprocessing failed")

        working_video = Path(temp_file.name)

    # Process video and store all frame differences
    cap = cv2.VideoCapture(str(working_video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {working_video}")

    working_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    working_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    image_size = working_width * working_height

    start_frame = int(start_offset * working_fps)
    frame_interval = max(1, int(round(sample_interval * working_fps)))
    min_frames_between = max(1, int(round(min_scene_len * working_fps)))

    # Compute all differences (this is the expensive part - do once!)
    frame_diffs = []  # List of (frame_num, normalized_diff)
    prev_frame_binary = None
    frame_num = start_frame

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        while frame_num < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if (frame_num - start_frame) % frame_interval != 0:
                frame_num += 1
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            if prev_frame_binary is not None:
                diff = np.abs(binary.astype(np.int16) - prev_frame_binary.astype(np.int16))
                non_zero_count = np.count_nonzero(diff)
                normalized_diff = non_zero_count / image_size
                frame_diffs.append((frame_num, normalized_diff))

            prev_frame_binary = binary
            frame_num += 1

    finally:
        cap.release()
        if temp_file is not None:
            try:
                os.unlink(temp_file.name)
            except Exception:
                pass

    logger.info(f"Computed {len(frame_diffs)} frame differences")

    # Step 2: Sweep thresholds on regular grid (e.g., 0.010, 0.011, 0.012, ...)
    thresholds = np.arange(threshold_range[0], threshold_range[1] + threshold_step, threshold_step)
    slide_counts = []

    for thresh in thresholds:
        count = 0
        last_change_frame = start_frame

        for frame_num, diff_val in frame_diffs:
            if diff_val >= thresh:
                if frame_num - last_change_frame >= min_frames_between:
                    count += 1
                    last_change_frame = frame_num

        slide_counts.append(count)
        logger.debug(f"Threshold {thresh:.3f}: {count} slides")

    # Step 3: Find flat region (stable slide count)
    # Look for longest plateau in the curve
    best_threshold_idx = _find_optimal_threshold_idx(slide_counts, thresholds)
    optimal_threshold = thresholds[best_threshold_idx]
    optimal_count = slide_counts[best_threshold_idx]

    logger.info(
        f"Adaptive threshold selected: {optimal_threshold:.3f} ({optimal_count} slides detected)"
    )

    # Step 4: Apply optimal threshold to get final timestamps
    timestamps = []
    last_change_frame = start_frame

    for frame_num, diff_val in frame_diffs:
        if diff_val >= optimal_threshold:
            if frame_num - last_change_frame >= min_frames_between:
                timestamp = frame_num / working_fps
                timestamps.append(timestamp)
                last_change_frame = frame_num

    return timestamps


def _find_optimal_threshold_idx(slide_counts: list[int], thresholds: np.ndarray) -> int:
    """Find optimal threshold index using flat region heuristic.

    Strategy: Find stable plateau with reasonable slide count.
    Prefer earlier thresholds (more sensitive) when plateaus have similar length.

    Args:
        slide_counts: Number of slides detected at each threshold.
        thresholds: Threshold values tested.

    Returns:
        Index of optimal threshold.
    """
    if len(slide_counts) < 3:
        # Not enough data points, prefer lowest threshold (most sensitive)
        return 0

    # Find all plateaus (consecutive indices with same or similar counts)
    plateaus = []  # List of (start_idx, length, avg_count)
    current_plateau_start = 0
    current_plateau_len = 1

    for i in range(1, len(slide_counts)):
        # Consider counts within 1 of each other as same plateau
        if abs(slide_counts[i] - slide_counts[i - 1]) <= 1:
            current_plateau_len += 1
        else:
            # Save current plateau
            avg_count = (
                sum(
                    slide_counts[
                        current_plateau_start : current_plateau_start + current_plateau_len
                    ]
                )
                / current_plateau_len
            )
            plateaus.append((current_plateau_start, current_plateau_len, avg_count))
            current_plateau_start = i
            current_plateau_len = 1

    # Add final plateau
    avg_count = (
        sum(slide_counts[current_plateau_start : current_plateau_start + current_plateau_len])
        / current_plateau_len
    )
    plateaus.append((current_plateau_start, current_plateau_len, avg_count))

    # Score plateaus: heavily prefer higher slide counts (avoid under-detection)
    # Score = slide_count^1.5 * length - prioritize slide count over stability
    best_plateau = max(plateaus, key=lambda p: (p[2] ** 1.5) * p[1] if p[2] > 0 else 0)

    best_start, best_len, best_avg = best_plateau

    # Return middle of best plateau
    optimal_idx = best_start + best_len // 2

    logger.debug(
        f"Found {len(plateaus)} plateaus, selected plateau at indices "
        f"{best_start}-{best_start + best_len - 1} (length={best_len}, avg_count={best_avg:.1f}), "
        f"selecting index {optimal_idx} (threshold={thresholds[optimal_idx]:.3f})"
    )

    return optimal_idx


def detect_slides_normalized(
    video_path: Path,
    start_offset: float = 3.0,
    min_scene_len: float = 2.0,
    z_threshold: float = 3.0,
    window_seconds: float = 7.0,
    sample_interval: float = 1.0,
    max_resolution: int = 360,
    target_fps: float = 5.0,
) -> list[float]:
    """Detect slide changes using rolling window normalization.

    This method addresses the heavy-tailed distribution problem by computing
    z-scores relative to a local rolling window. This makes the threshold
    video-independent and robust to varying baseline noise levels.

    Method:
        1. Compute raw frame differences (median pixel difference)
        2. For each frame, compute rolling window statistics:
           - median m_t over window [t-W, t)
           - MAD (median absolute deviation) s_t over window
        3. Normalize: z_t = (d_t - m_t) / (s_t + epsilon)
        4. Threshold on z-score (typically z > 3.0 for outliers)

    Args:
        video_path: Path to video file.
        start_offset: Skip first N seconds.
        min_scene_len: Minimum scene length in seconds (refractory period).
        z_threshold: Z-score threshold for detection (default 3.0).
                    Typical range: 2.5-4.0. Higher = less sensitive.
        window_seconds: Rolling window size in seconds (default 7.0).
                       Typical range: 5-10 seconds.
        sample_interval: Time between compared frames.
        max_resolution: Maximum resolution for processing.
        target_fps: Target frame rate for processing.

    Returns:
        List of timestamps where slide changes occur.
    """
    # Preprocess video if needed
    temp_handle = None
    try:
        working_path, temp_handle, working_fps = _preprocess_video_if_needed(
            video_path, max_resolution, target_fps
        )

        cap = cv2.VideoCapture(str(working_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {working_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = working_fps
        working_fps = fps

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = int(start_offset * working_fps)
        frame_interval = max(1, int(round(sample_interval * working_fps)))
        min_frames_between = max(1, int(round(min_scene_len * working_fps)))
        window_frames = max(1, int(round(window_seconds * working_fps)))

        # Compute all raw frame differences first
        frame_diffs = []  # List of (frame_num, raw_diff)
        prev_frame_binary = None
        frame_num = start_frame

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        logger.info(f"Computing frame differences (fps={working_fps:.1f})...")

        while frame_num < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if (frame_num - start_frame) % frame_interval != 0:
                frame_num += 1
                continue

            # Convert to grayscale and binarize
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            if prev_frame_binary is not None:
                diff = np.abs(binary.astype(np.int16) - prev_frame_binary.astype(np.int16))
                raw_diff = np.count_nonzero(diff) / binary.size
                frame_diffs.append((frame_num, raw_diff))

            prev_frame_binary = binary
            frame_num += 1

        cap.release()

        if len(frame_diffs) == 0:
            logger.warning("No frames to process")
            return []

        logger.info(f"Computed {len(frame_diffs)} frame differences")

        # Compute rolling window z-scores
        diffs_array = np.array([d for _, d in frame_diffs])
        z_scores = np.zeros(len(diffs_array))

        logger.info(f"Computing z-scores with {window_seconds}s window...")

        for i in range(len(diffs_array)):
            # Window: [max(0, i-W), i)
            window_start = max(0, i - window_frames)
            window = diffs_array[window_start:i]

            if len(window) < 2:
                z_scores[i] = 0.0
                continue

            # Compute robust statistics
            m_t = np.median(window)
            mad = np.median(np.abs(window - m_t))
            s_t = 1.4826 * mad  # Scale factor for MAD to approximate std

            # Normalize with small epsilon to avoid division by zero
            epsilon = 1e-6
            z_scores[i] = (diffs_array[i] - m_t) / (s_t + epsilon)

        # Detect slides using z-score threshold + refractory period
        timestamps = []
        last_change_frame = start_frame - min_frames_between  # Allow first detection

        logger.info(f"Detecting slides with z-threshold={z_threshold:.1f}...")

        for i, (frame_num, raw_diff) in enumerate(frame_diffs):
            z_score = z_scores[i]

            if z_score >= z_threshold:
                if frame_num - last_change_frame >= min_frames_between:
                    timestamp = frame_num / working_fps
                    timestamps.append(timestamp)
                    last_change_frame = frame_num

                    logger.debug(
                        f"Slide change at {timestamp:.2f}s "
                        f"(frame {frame_num}, z={z_score:.2f}, raw={raw_diff:.4f})"
                    )

    finally:
        # Clean up temp file if created
        if temp_handle is not None:
            try:
                os.unlink(temp_handle.name)
                logger.debug(f"Cleaned up temporary file {temp_handle.name}")
            except Exception:
                pass

    logger.info(f"Found {len(timestamps)} slide changes")
    return timestamps
