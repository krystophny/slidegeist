"""Test slide detection with labeled lecture videos.

Videos are downloaded once and cached in /tmp/slidegeist_labeled_videos/
"""

import subprocess
from pathlib import Path
from typing import NamedTuple

import pytest


class LabeledVideo(NamedTuple):
    """Lecture video with known slide count."""

    name: str
    url: str
    slide_count: int
    notes: str = ""


# Test videos with known slide counts
LABELED_VIDEOS = [
    LabeledVideo(
        name="tugraz_presentation_15",
        url="https://tube.tugraz.at/portal/watch/53ceb6bf-bc16-4212-8fe6-a6080f7e3d79",
        slide_count=15,
        notes="TU Graz presentation with 15 labeled slides",
    ),
    LabeledVideo(
        name="tugraz_presentation_15b",
        url="https://tube.tugraz.at/portal/watch/58502d58-ba76-4628-ac55-f7cca0584d10",
        slide_count=15,
        notes="TU Graz presentation with 15 labeled slides (second sample)",
    ),
    LabeledVideo(
        name="tugraz_presentation_30",
        url="https://tube.tugraz.at/portal/watch/4bb83577-3a97-4832-90ac-3d72600336a4",
        slide_count=30,
        notes="TU Graz presentation with 30 labeled slides",
    ),
]


def get_cache_dir() -> Path:
    """Get cache directory for test videos."""
    cache_dir = Path("/tmp/slidegeist_labeled_videos")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_video_path(video: LabeledVideo) -> Path:
    """Get path to cached video."""
    return get_cache_dir() / f"{video.name}.mp4"


def download_video(video: LabeledVideo) -> Path:
    """Download video if not cached.

    Args:
        video: Video information.

    Returns:
        Path to video file.
    """
    video_path = get_video_path(video)

    if video_path.exists():
        print(f"Using cached video: {video_path}")
        return video_path

    print(f"Downloading {video.name} from {video.url}")

    cmd = ["yt-dlp", "-f", "best[ext=mp4]", "-o", str(video_path), video.url]

    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=600)
        print(f"Downloaded to {video_path}")
        return video_path
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Download timeout for {video.name}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Download failed: {e.stderr.decode()}")


def run_detection(video_path: Path, threshold: float) -> int:
    """Run slide detection and return slide count.

    Args:
        video_path: Path to video.
        threshold: Detection threshold.

    Returns:
        Number of slides detected.
    """
    import tempfile

    output_dir = Path(tempfile.mkdtemp())

    try:
        cmd = [
            "slidegeist",
            "slides",
            str(video_path),
            "--out",
            str(output_dir),
            "--scene-threshold",
            str(threshold),
        ]

        subprocess.run(cmd, check=True, capture_output=True, timeout=300)

        slides = list(output_dir.glob("slide_*.jpg"))
        return len(slides)

    finally:
        import shutil

        shutil.rmtree(output_dir, ignore_errors=True)


@pytest.mark.manual
@pytest.mark.parametrize("video", LABELED_VIDEOS, ids=lambda v: v.name)
def test_labeled_video_default_threshold(video: LabeledVideo):
    """Test that default threshold detects correct number of slides.

    Run with: pytest -v -m manual tests/test_labeled_videos.py
    """
    video_path = download_video(video)

    # Test with default threshold (0.03)
    detected = run_detection(video_path, 0.03)

    print(f"\n{video.name}:")
    print(f"  Expected: {video.slide_count} slides")
    print(f"  Detected: {detected} slides (threshold=0.03)")
    print(f"  Status: {'✓ PASS' if detected >= video.slide_count else '✗ FAIL'}")

    # We need at least the labeled slide count
    # Allow some over-detection (up to 50% more) but not under-detection
    assert detected >= video.slide_count, (
        f"Under-detected slides: got {detected}, need at least {video.slide_count}"
    )
    assert detected <= video.slide_count * 1.5, (
        f"Excessive over-detection: got {detected}, expected ~{video.slide_count}"
    )


@pytest.mark.manual
@pytest.mark.parametrize("video", LABELED_VIDEOS, ids=lambda v: v.name)
def test_labeled_video_threshold_sweep(video: LabeledVideo):
    """Test multiple thresholds to find optimal range.

    Run with: pytest -v -s -m manual tests/test_labeled_videos.py::test_labeled_video_threshold_sweep
    """
    video_path = download_video(video)

    print(f"\n{'=' * 60}")
    print(f"Threshold Sweep: {video.name}")
    print(f"Expected slides: {video.slide_count}")
    print(f"{'=' * 60}\n")

    thresholds = [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.10]
    results = []

    for threshold in thresholds:
        detected = run_detection(video_path, threshold)
        results.append((threshold, detected))

        status = "✓" if detected >= video.slide_count else "✗"
        print(f"  {threshold:.3f}: {detected:3d} slides {status}")

    print(f"\n{'=' * 60}")

    # Find best threshold (closest to expected without under-detecting)
    valid_results = [(t, d) for t, d in results if d >= video.slide_count]

    if valid_results:
        best_threshold, best_count = min(valid_results, key=lambda x: abs(x[1] - video.slide_count))
        print(f"Best threshold: {best_threshold:.3f} ({best_count} slides)")
    else:
        print("WARNING: No threshold detected minimum required slides!")

    # At least one threshold should work
    assert len(valid_results) > 0, f"No threshold detected at least {video.slide_count} slides"
