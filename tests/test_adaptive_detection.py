"""Test adaptive threshold detection."""

from pathlib import Path

import pytest

from slidegeist.pixel_diff_detector import detect_slides_adaptive


@pytest.mark.manual
def test_adaptive_detection_tugraz():
    """Test adaptive detection on TU Graz 15-slide video.

    Run with: pytest -v -s -m manual tests/test_adaptive_detection.py
    """
    video_path = Path("/tmp/slidegeist_labeled_videos/tugraz_presentation.mp4")

    if not video_path.exists():
        pytest.skip("TU Graz video not cached, run test_labeled_videos.py first")

    # Run adaptive detection
    timestamps = detect_slides_adaptive(video_path)

    num_slides = len(timestamps) + 1  # +1 because timestamps are transitions
    print("\nAdaptive detection results:")
    print(f"  Detected: {num_slides} slides")
    print("  Expected: 15 slides")
    print(f"  Transitions: {timestamps}")

    # Should detect at least 15 slides (maybe a few more is OK)
    assert num_slides >= 15, f"Under-detected: {num_slides} < 15"
    assert num_slides <= 20, f"Over-detected: {num_slides} > 20"
