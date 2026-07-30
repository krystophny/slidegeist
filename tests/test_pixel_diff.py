"""Tests for the pixel-diff detector utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import pytest

from slidegeist.pixel_diff_detector import detect_slides_pixel_diff


def test_detect_slides_pixel_diff_fallbacks_to_default_fps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Videos reporting 0 FPS should trigger a safe fallback."""
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"fake")

    class DummyCapture:
        def __init__(self, _: Any) -> None:
            self.props = {
                cv2.CAP_PROP_FRAME_HEIGHT: 240.0,
                cv2.CAP_PROP_FRAME_WIDTH: 320.0,
                cv2.CAP_PROP_FPS: 0.0,
                cv2.CAP_PROP_FRAME_COUNT: 0.0,
            }

        def get(self, prop: int) -> float:
            return float(self.props.get(prop, 0.0))

        def release(self) -> None:
            return None

        def isOpened(self) -> bool:  # noqa: N802 - mirrors cv2.VideoCapture
            return True

        def set(self, *_: Any, **__: Any) -> None:
            return None

        def read(self) -> tuple[bool, Any]:
            return False, None

    monkeypatch.setattr("cv2.VideoCapture", lambda _: DummyCapture(None))

    timestamps = detect_slides_pixel_diff(
        video_path,
        start_offset=0.0,
        min_scene_len=1.0,
        threshold=0.1,
        sample_interval=1.0,
        max_resolution=360,
        target_fps=120.0,
    )

    assert timestamps == []
