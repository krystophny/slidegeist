#!/usr/bin/env python3
"""Diagnostic script to plot threshold sweep for a video.

Usage:
    python scripts/plot_threshold_sweep.py /path/to/video.mp4
    python scripts/plot_threshold_sweep.py /path/to/video.mp4 --expected-slides 15
    python scripts/plot_threshold_sweep.py /path/to/video.mp4 --compare-pyscenedetect
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

# Import labeled videos from test file
sys.path.insert(0, str(Path(__file__).parent.parent))
from tests.test_labeled_videos import LABELED_VIDEOS, download_video

# Minimal plotting - works without matplotlib
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed - will only show text output")
    print("Install with: pip install matplotlib")

# Optional PySceneDetect
try:
    from scenedetect import ContentDetector, SceneManager, open_video
    from scenedetect.detectors import HashDetector, HistogramDetector
    from scenedetect.stats_manager import StatsManager
    HAS_PYSCENEDETECT = True
except ImportError:
    HAS_PYSCENEDETECT = False


def compute_frame_diffs(video_path: Path) -> tuple[list[tuple[int, float]], float]:
    """Compute all frame differences for a video.

    Returns:
        (frame_diffs, working_fps) where frame_diffs is list of (frame_num, diff_value)
    """
    from slidegeist.pixel_diff_detector import cv2, np, os, subprocess, tempfile

    # Simplified version of preprocessing from detect_slides_adaptive
    cap = cv2.VideoCapture(str(video_path))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fps <= 0:
        fps = 30.0

    # Preprocess if needed
    max_resolution = 360
    target_fps = 5.0
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
            filters.append(f'scale={new_width}:{new_height}')
        if fps > target_fps:
            fps_ratio = int(round(fps / target_fps))
            actual_fps = fps / fps_ratio
            filters.append(f'fps=fps={actual_fps}')
            working_fps = actual_fps

        filter_str = ','.join(filters)

        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_file.close()

        print(f"Preprocessing video: {width}x{height}@{fps:.1f}fps -> {new_width}x{new_height}@{working_fps:.1f}fps")
        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-vf', filter_str,
            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '28',
            '-y', temp_file.name
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        working_video = Path(temp_file.name)

    # Compute frame differences
    cap = cv2.VideoCapture(str(working_video))
    working_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    working_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    image_size = working_width * working_height

    start_offset = 3.0
    sample_interval = 1.0
    start_frame = int(start_offset * working_fps)
    frame_interval = max(1, int(round(sample_interval * working_fps)))

    frame_diffs = []
    prev_frame_binary = None
    frame_num = start_frame

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        print("Computing frame differences...")

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

    print(f"Computed {len(frame_diffs)} frame differences")
    return frame_diffs, working_fps


def sweep_thresholds(
    frame_diffs: list[tuple[int, float]],
    working_fps: float,
    threshold_range: tuple[float, float] = (0.01, 0.10),
    threshold_step: float = 0.001,
    min_scene_len: float = 2.0,
    start_offset: float = 3.0
) -> tuple[np.ndarray, np.ndarray]:
    """Sweep thresholds and count slides at each.

    Returns:
        (thresholds, slide_counts)
    """
    thresholds = np.arange(threshold_range[0], threshold_range[1] + threshold_step, threshold_step)
    slide_counts = []

    start_frame = int(start_offset * working_fps)
    min_frames_between = max(1, int(round(min_scene_len * working_fps)))

    print(f"Sweeping {len(thresholds)} thresholds from {threshold_range[0]:.3f} to {threshold_range[1]:.3f}...")

    for thresh in thresholds:
        count = 0
        last_change_frame = start_frame

        for frame_num, diff_val in frame_diffs:
            if diff_val >= thresh:
                if frame_num - last_change_frame >= min_frames_between:
                    count += 1
                    last_change_frame = frame_num

        slide_counts.append(count)

    return thresholds, np.array(slide_counts)


def compute_z_scores(
    frame_diffs: list[tuple[int, float]],
    window_seconds: float = 7.0,
    working_fps: float = 5.0
) -> np.ndarray:
    """Compute rolling window z-scores for frame differences.

    Args:
        frame_diffs: List of (frame_num, diff_value) tuples.
        window_seconds: Rolling window size in seconds.
        working_fps: Frame rate of the video.

    Returns:
        Array of z-scores, same length as frame_diffs.
    """
    diffs_array = np.array([d for _, d in frame_diffs])
    z_scores = np.zeros(len(diffs_array))
    window_frames = int(window_seconds * working_fps)

    print(f"Computing z-scores with {window_seconds}s window ({window_frames} frames)...")

    for i in range(len(diffs_array)):
        # Window: [max(0, i-W), i)
        window_start = max(0, i - window_frames)
        window = diffs_array[window_start:i]

        if len(window) < 2:
            z_scores[i] = 0.0
            continue

        # Compute robust statistics using MAD
        m_t = np.median(window)
        mad = np.median(np.abs(window - m_t))
        s_t = 1.4826 * mad  # Scale factor for MAD to approximate std

        # Normalize with small epsilon to avoid division by zero
        epsilon = 1e-6
        z_scores[i] = (diffs_array[i] - m_t) / (s_t + epsilon)

    return z_scores


def sweep_z_thresholds(
    frame_diffs: list[tuple[int, float]],
    z_scores: np.ndarray,
    working_fps: float,
    z_threshold_range: tuple[float, float] = (2.0, 5.0),
    z_threshold_step: float = 0.1,
    min_scene_len: float = 2.0,
    start_offset: float = 3.0
) -> tuple[np.ndarray, np.ndarray]:
    """Sweep over z-score thresholds and count detections.

    Args:
        frame_diffs: List of (frame_num, diff_value) tuples.
        z_scores: Array of z-scores for each frame.
        working_fps: Frame rate of the video.
        z_threshold_range: (min, max) z-score thresholds to test.
        z_threshold_step: Step size for z-threshold sweep.
        min_scene_len: Minimum scene length in seconds (refractory period).
        start_offset: Offset in seconds to start detection.

    Returns:
        Tuple of (z_thresholds, slide_counts) arrays.
    """
    min_frames_between = int(min_scene_len * working_fps)
    start_frame = int(start_offset * working_fps)

    z_min, z_max = z_threshold_range
    z_thresholds = np.arange(z_min, z_max + z_threshold_step, z_threshold_step)
    slide_counts = []

    print(f"Sweeping {len(z_thresholds)} z-thresholds from {z_min:.1f} to {z_max:.1f}...")

    for z_thresh in z_thresholds:
        count = 0
        last_change_frame = start_frame - min_frames_between

        for i, (frame_num, _) in enumerate(frame_diffs):
            if z_scores[i] >= z_thresh:
                if frame_num - last_change_frame >= min_frames_between:
                    count += 1
                    last_change_frame = frame_num

        slide_counts.append(count)

    return z_thresholds, np.array(slide_counts)


def compute_pyscenedetect_scores(video_path: Path, detector_name: str) -> tuple[list[float], float, str]:
    """Compute PySceneDetect detector scores using stats file.

    Args:
        video_path: Path to video
        detector_name: 'content', 'histogram', or 'hash'

    Returns:
        (score_vals, fps, metric_key) where metric_key is the CSV column name
    """
    if not HAS_PYSCENEDETECT:
        raise RuntimeError("PySceneDetect not installed")

    # Create detector and determine metric key prefix
    # Note: actual CSV column names have parameters in brackets
    if detector_name == 'content':
        detector = ContentDetector()
        metric_prefix = 'content_val'
        print("\nRunning PySceneDetect ContentDetector (HSV)...")
    elif detector_name == 'histogram':
        detector = HistogramDetector()
        metric_prefix = 'hist_diff'
        print("\nRunning PySceneDetect HistogramDetector (YUV)...")
    elif detector_name == 'hash':
        detector = HashDetector()
        metric_prefix = 'hash_dist'
        print("\nRunning PySceneDetect HashDetector (perceptual hash)...")
    else:
        raise ValueError(f"Unknown detector: {detector_name}")

    # Create temp stats file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        stats_file = Path(f.name)

    video = None
    try:
        # Process video with detector and save stats
        video = open_video(str(video_path))
        fps = video.frame_rate

        stats_manager = StatsManager()
        scene_manager = SceneManager(stats_manager)
        scene_manager.add_detector(detector)

        # Process all frames
        scene_manager.detect_scenes(video)

        # Save stats to file
        stats_manager.save_to_csv(str(stats_file))

        # Read stats file - find column matching metric prefix
        import csv
        score_vals = []
        with open(stats_file) as f:
            reader = csv.DictReader(f)
            # Find actual column name (may have parameters in brackets)
            if reader.fieldnames:
                metric_key = None
                for field in reader.fieldnames:
                    if field.startswith(metric_prefix):
                        metric_key = field
                        break

                if not metric_key:
                    raise RuntimeError(f"No column found starting with {metric_prefix}")

                for row in reader:
                    if metric_key in row and row[metric_key]:
                        score_vals.append(float(row[metric_key]))

        print(f"PySceneDetect {detector_name} computed {len(score_vals)} frame scores (column: {metric_key})")
        return score_vals, fps, metric_key

    finally:
        # Clean up video
        if video is not None:
            try:
                del video
            except Exception:
                pass

        # Clean up temp file
        try:
            stats_file.unlink()
        except Exception:
            pass


def sweep_pyscenedetect_thresholds(
    content_vals: list[float],
    fps: float,
    threshold_range: tuple[float, float] = (1.0, 25.0),
    threshold_step: float = 0.5,
    min_scene_len: float = 2.0,
    start_offset: float = 3.0
) -> tuple[np.ndarray, np.ndarray]:
    """Sweep PySceneDetect thresholds on pre-computed scores.

    Returns:
        (thresholds, scene_counts)
    """
    thresholds = np.arange(threshold_range[0], threshold_range[1] + threshold_step, threshold_step)
    scene_counts = []

    start_frame = int(start_offset * fps)
    min_frames_between = max(1, int(round(min_scene_len * fps)))

    print(f"Sweeping {len(thresholds)} PySceneDetect thresholds from {threshold_range[0]:.1f} to {threshold_range[1]:.1f}...")

    for thresh in thresholds:
        count = 0
        last_cut_frame = start_frame

        for frame_num, score in enumerate(content_vals, start=start_frame):
            if score >= thresh:
                if frame_num - last_cut_frame >= min_frames_between:
                    count += 1
                    last_cut_frame = frame_num

        scene_counts.append(count)

    return thresholds, np.array(scene_counts)


def plot_sweep(
    thresholds: np.ndarray,
    slide_counts: np.ndarray,
    expected_slides: int | None = None,
    video_name: str = "Video",
    pyscene_results: list[tuple[str, np.ndarray, np.ndarray]] | None = None,
    z_score_results: tuple[np.ndarray, np.ndarray] | None = None,
    opencast_result: tuple[float, int] | None = None
):
    """Plot threshold sweep results.

    Args:
        thresholds: slidegeist thresholds
        slide_counts: slidegeist slide counts
        expected_slides: expected slide count
        video_name: video filename
        pyscene_results: list of (detector_name, thresholds, counts) tuples
        z_score_results: tuple of (z_thresholds, z_slide_counts)
        opencast_result: tuple of (opencast_threshold, opencast_count) if available
    """
    if not HAS_MATPLOTLIB:
        print("\nCannot plot without matplotlib")
        return

    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot our method (slidegeist) - normalize to 0-1 like others
    thresh_min_sg = thresholds.min()
    thresh_max_sg = thresholds.max()
    normalized_thresholds_sg = (thresholds - thresh_min_sg) / (thresh_max_sg - thresh_min_sg)

    ax.plot(normalized_thresholds_sg, slide_counts, 'b-', linewidth=2.5,
            label=f'slidegeist (binary pixel diff, {thresh_min_sg:.2f}-{thresh_max_sg:.2f})', zorder=10)
    ax.scatter(normalized_thresholds_sg, slide_counts, c='blue', s=30, alpha=0.6, zorder=10)

    # Plot PySceneDetect methods if available - normalize their thresholds to 0-1
    colors = ['red', 'orange', 'purple']
    if pyscene_results:
        for idx, (detector_name, pyscene_thresholds, pyscene_counts) in enumerate(pyscene_results):
            color = colors[idx % len(colors)]

            # Normalize thresholds to 0-1 range
            thresh_min = pyscene_thresholds.min()
            thresh_max = pyscene_thresholds.max()
            normalized_thresholds = (pyscene_thresholds - thresh_min) / (thresh_max - thresh_min)

            label_map = {
                'content': f'PySceneDetect ContentDetector (HSV, {thresh_min:.1f}-{thresh_max:.1f})',
                'histogram': f'PySceneDetect HistogramDetector (YUV, {thresh_min:.1f}-{thresh_max:.1f})',
                'hash': f'PySceneDetect HashDetector (pHash, {thresh_min:.2f}-{thresh_max:.2f})'
            }
            label = label_map.get(detector_name, detector_name)

            ax.plot(normalized_thresholds, pyscene_counts, color=color, linestyle='-',
                    linewidth=2, label=label, alpha=0.8)
            ax.scatter(normalized_thresholds, pyscene_counts, c=color, s=20, alpha=0.5)

    # Plot z-score results if available
    if z_score_results:
        z_thresholds, z_slide_counts = z_score_results
        # Normalize z-thresholds to 0-1 range
        z_min = z_thresholds.min()
        z_max = z_thresholds.max()
        normalized_z_thresholds = (z_thresholds - z_min) / (z_max - z_min)

        ax.plot(normalized_z_thresholds, z_slide_counts, color='purple', linestyle='-',
                linewidth=2.5, label=f'slidegeist (z-score, {z_min:.1f}-{z_max:.1f})', alpha=0.9)
        ax.scatter(normalized_z_thresholds, z_slide_counts, c='purple', s=30, alpha=0.6)

    # Mark expected slide count if provided
    if expected_slides:
        ax.axhline(y=expected_slides, color='g', linestyle='--', linewidth=2,
                   label=f'Expected: {expected_slides} slides')
        # Shade acceptable range (±20%)
        lower = expected_slides * 0.8
        upper = expected_slides * 1.2
        ax.axhspan(lower, upper, alpha=0.1, color='green', label='±20% range')

        # Mark optimal thresholds for each method
        # Find where each method crosses expected_slides
        # slidegeist
        diffs_sg = np.abs(slide_counts - expected_slides)
        if diffs_sg.min() <= 2:  # Within 2 slides
            idx_sg = np.argmin(diffs_sg)
            opt_thresh_sg = normalized_thresholds_sg[idx_sg]
            ax.plot(opt_thresh_sg, slide_counts[idx_sg], 'b*', markersize=15,
                   markeredgecolor='white', markeredgewidth=1.5, zorder=20,
                   label='slidegeist optimal')

        # PySceneDetect methods
        if pyscene_results:
            colors_marker = ['red', 'orange', 'purple']
            for idx, (detector_name, pyscene_thresholds, pyscene_counts) in enumerate(pyscene_results):
                diffs = np.abs(pyscene_counts - expected_slides)
                if diffs.min() <= 2:
                    idx_opt = np.argmin(diffs)
                    # Normalize the optimal threshold
                    thresh_min = pyscene_thresholds.min()
                    thresh_max = pyscene_thresholds.max()
                    opt_thresh_norm = (pyscene_thresholds[idx_opt] - thresh_min) / (thresh_max - thresh_min)

                    ax.plot(opt_thresh_norm, pyscene_counts[idx_opt], '*',
                           color=colors_marker[idx % len(colors_marker)], markersize=15,
                           markeredgecolor='white', markeredgewidth=1.5, zorder=20)

        # Z-score method optimal marker
        if z_score_results:
            z_thresholds, z_slide_counts = z_score_results
            diffs_z = np.abs(z_slide_counts - expected_slides)
            if diffs_z.min() <= 2:
                idx_z = np.argmin(diffs_z)
                z_min = z_thresholds.min()
                z_max = z_thresholds.max()
                opt_thresh_z_norm = (z_thresholds[idx_z] - z_min) / (z_max - z_min)
                ax.plot(opt_thresh_z_norm, z_slide_counts[idx_z], '*',
                       color='purple', markersize=15,
                       markeredgecolor='white', markeredgewidth=1.5, zorder=20,
                       label='z-score optimal')

        # Opencast-style optimization marker
        if opencast_result:
            opencast_threshold, opencast_count = opencast_result
            # Normalize opencast threshold to our raw threshold scale
            thresh_min_sg = thresholds.min()
            thresh_max_sg = thresholds.max()
            opencast_norm = (opencast_threshold - thresh_min_sg) / (thresh_max_sg - thresh_min_sg)
            ax.plot(opencast_norm, opencast_count, 'D',
                   color='green', markersize=12,
                   markeredgecolor='white', markeredgewidth=1.5, zorder=20,
                   label=f'Opencast-style (adapted, θ={opencast_threshold:.3f})')

    ax.set_xlabel('Normalized Threshold (log scale)', fontsize=12)
    ax.set_ylabel('Number of Slides (log scale)', fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(f'Threshold Sweep Comparison: {video_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()

    # Save plot with video-specific filename
    output_path = Path(f'threshold_sweep_{video_name}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # Try to display
    try:
        plt.show()
    except Exception:
        print("(Cannot display plot - saved to file)")


def print_text_summary(
    thresholds: np.ndarray,
    slide_counts: np.ndarray,
    expected_slides: int | None = None
):
    """Print text summary of sweep results."""
    print("\n" + "="*80)
    print("THRESHOLD SWEEP RESULTS")
    print("="*80)
    print(f"\n{'Threshold':>10} | {'Slides':>6} | {'Change':>7} | Status")
    print("-"*80)

    # Show every 5th point to avoid too much output
    step = max(1, len(thresholds) // 20)

    for i in range(0, len(thresholds), step):
        thresh = thresholds[i]
        count = slide_counts[i]
        change = "" if i == 0 else f"{count - slide_counts[i-step]:+3d}"

        if expected_slides:
            if expected_slides * 0.8 <= count <= expected_slides * 1.2:
                status = "✓ good"
            elif count < expected_slides:
                status = f"✗ under ({expected_slides - count} missing)"
            else:
                status = f"✗ over (+{count - expected_slides} extra)"
        else:
            status = ""

        print(f"{thresh:10.3f} | {count:6d} | {change:>7} | {status}")

    print("\n" + "="*80)

    # Find best thresholds
    if expected_slides:
        # Find threshold closest to expected count
        diffs = np.abs(slide_counts - expected_slides)
        best_idx = np.argmin(diffs)

        print(f"\nClosest to {expected_slides} slides:")
        print(f"  Threshold: {thresholds[best_idx]:.3f}")
        print(f"  Slides: {slide_counts[best_idx]}")
        print(f"  Error: {slide_counts[best_idx] - expected_slides:+d} slides")


def main():
    parser = argparse.ArgumentParser(
        description="Plot threshold sweep diagnostic for slide detection"
    )
    parser.add_argument(
        "video",
        nargs="?",
        help="Video name from labeled list OR path to video file (optional if --all is used)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate plots for all labeled videos"
    )
    parser.add_argument(
        "--expected-slides", "-e", type=int,
        help="Expected number of slides (for comparison, only used with explicit path)"
    )
    parser.add_argument(
        "--threshold-min", type=float, default=0.01,
        help="Minimum threshold for slidegeist (default: 0.01)"
    )
    parser.add_argument(
        "--threshold-max", type=float, default=0.10,
        help="Maximum threshold for slidegeist (default: 0.10)"
    )
    parser.add_argument(
        "--threshold-step", type=float, default=0.001,
        help="Threshold step size for slidegeist (default: 0.001)"
    )
    parser.add_argument(
        "--compare-pyscenedetect", action="store_true",
        help="Also run PySceneDetect ContentDetector for comparison"
    )
    parser.add_argument(
        "--pyscene-threshold-min", type=float, default=1.0,
        help="Minimum threshold for PySceneDetect (default: 1.0)"
    )
    parser.add_argument(
        "--pyscene-threshold-max", type=float, default=25.0,
        help="Maximum threshold for PySceneDetect (default: 25.0)"
    )
    parser.add_argument(
        "--pyscene-threshold-step", type=float, default=0.5,
        help="Threshold step size for PySceneDetect (default: 0.5)"
    )

    args = parser.parse_args()

    # Determine which videos to process
    videos_to_process = []

    if args.all:
        # Process all labeled videos
        for labeled_video in LABELED_VIDEOS:
            video_path = download_video(labeled_video)
            videos_to_process.append((video_path, labeled_video.slide_count, labeled_video.name))
    elif args.video:
        # Check if video is a labeled video name
        labeled_video = next((v for v in LABELED_VIDEOS if v.name == args.video), None)
        if labeled_video:
            video_path = download_video(labeled_video)
            videos_to_process.append((video_path, labeled_video.slide_count, labeled_video.name))
        else:
            # Treat as file path
            video_path = Path(args.video)
            if not video_path.exists():
                print(f"Error: Video not found: {video_path}")
                sys.exit(1)
            videos_to_process.append((video_path, args.expected_slides, video_path.stem))
    else:
        print("Error: Must provide video name/path or use --all")
        parser.print_help()
        sys.exit(1)

    # Process each video
    for video_path, expected_slides, video_name in videos_to_process:
        print(f"\n{'='*80}")
        print(f"Processing: {video_name}")
        print(f"{'='*80}\n")

        # Compute frame differences for slidegeist
        frame_diffs, working_fps = compute_frame_diffs(video_path)

        # Sweep thresholds for slidegeist
        global threshold_range
        threshold_range = (args.threshold_min, args.threshold_max)
        thresholds, slide_counts = sweep_thresholds(
            frame_diffs,
            working_fps,
            threshold_range=threshold_range,
            threshold_step=args.threshold_step
        )

        # Print text summary for slidegeist
        print_text_summary(thresholds, slide_counts, expected_slides)

        # Compute z-scores and sweep z-thresholds
        print("\nComputing rolling window z-scores...")
        z_scores = compute_z_scores(frame_diffs, window_seconds=7.0, working_fps=working_fps)
        z_thresholds, z_slide_counts = sweep_z_thresholds(
            frame_diffs,
            z_scores,
            working_fps,
            z_threshold_range=(2.0, 5.0),
            z_threshold_step=0.1
        )
        print(f"Z-score sweep complete: {len(z_thresholds)} thresholds tested")

        # Run Opencast-style optimization if expected_slides is known
        opencast_result = None
        if expected_slides:
            print(f"\nRunning Opencast FFmpeg-based optimization (target={expected_slides})...")
            from slidegeist.ffmpeg_scene import detect_scenes_opencast
            try:
                # NOTE: Using stability_threshold=2.0s instead of Opencast's default 60.0s
                # Rationale: Slide presentations have much faster scene changes than
                # general lecture videos. A 60s threshold would merge all slides together.
                # For slide detection, 2s is appropriate to filter camera/presenter movements
                # while preserving slide transitions.
                opencast_timestamps, opencast_threshold = detect_scenes_opencast(
                    video_path,
                    target_segments=expected_slides,
                    max_cycles=3,
                    initial_threshold=0.025,
                    stability_threshold=2.0,  # Adapted for slide detection (not Opencast's 60s)
                    start_offset=3.0
                )
                # Count segments: N cut points = N+1 segments
                opencast_segments = len(opencast_timestamps) + 1 if opencast_timestamps else 1
                opencast_result = (opencast_threshold, opencast_segments)
                print(f"Opencast-style optimization: threshold={opencast_threshold:.4f}, segments={opencast_segments}")
            except Exception as e:
                print(f"Warning: Opencast-style optimization failed: {e}")

        # Optionally run PySceneDetect comparison
        pyscene_results = []
        if args.compare_pyscenedetect:
            if not HAS_PYSCENEDETECT:
                print("\nWarning: PySceneDetect not installed. Install with: pip install scenedetect[opencv]")
            else:
                # Define detector configs: (name, threshold_range, step)
                detector_configs = [
                    ('content', (args.pyscene_threshold_min, args.pyscene_threshold_max), args.pyscene_threshold_step),
                    ('hash', (0.10, 0.50), 0.01),
                ]

                for detector_name, threshold_range_det, threshold_step in detector_configs:
                    try:
                        score_vals, pyscene_fps, metric_key = compute_pyscenedetect_scores(video_path, detector_name)
                        pyscene_thresholds, pyscene_counts = sweep_pyscenedetect_thresholds(
                            score_vals,
                            pyscene_fps,
                            threshold_range=threshold_range_det,
                            threshold_step=threshold_step
                        )
                        pyscene_results.append((detector_name, pyscene_thresholds, pyscene_counts))
                        print(f"PySceneDetect {detector_name} sweep complete: {len(pyscene_thresholds)} thresholds tested")
                    except Exception as e:
                        print(f"\nWarning: PySceneDetect {detector_name} comparison failed: {e}")

        # Plot if matplotlib available
        if HAS_MATPLOTLIB:
            plot_sweep(
                thresholds,
                slide_counts,
                expected_slides,
                video_name,
                pyscene_results if pyscene_results else None,
                z_score_results=(z_thresholds, z_slide_counts),
                opencast_result=opencast_result
            )
        else:
            print("\nInstall matplotlib to see plot: pip install matplotlib")


if __name__ == "__main__":
    main()
