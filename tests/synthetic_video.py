"""Independent synthetic oracle for slide-transition tests and benchmarks."""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np


def make_slide_video(
    path: Path,
    transitions: list[float],
    *,
    duration: float = 16.0,
    fps: int = 10,
    moving_occluder: bool = True,
    luminance_flicker_at: float | None = 6.2,
) -> None:
    """Write a variable-cadence lecture video with known transition times."""
    width, height = 640, 360
    codec = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), codec, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create synthetic AVI fixture")

    palette = [
        (246, 246, 246),
        (236, 246, 252),
        (248, 240, 232),
        (234, 248, 238),
        (244, 236, 250),
    ]
    total_frames = round(duration * fps)
    try:
        for frame_number in range(total_frames):
            timestamp = frame_number / fps
            slide_index = sum(timestamp >= transition for transition in transitions)
            frame = np.full((height, width, 3), palette[slide_index], dtype=np.uint8)
            cv2.rectangle(frame, (24, 20), (616, 334), (80, 80, 80), 2)
            cv2.putText(
                frame,
                f"PLASMA PHYSICS — SLIDE {slide_index + 1}",
                (48, 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (15, 15, 15),
                2,
                cv2.LINE_AA,
            )
            for line in range(slide_index + 2):
                y = 125 + 37 * line
                cv2.line(frame, (65, y), (300 + 45 * slide_index, y), (35, 35, 35), 4)
            cv2.circle(
                frame,
                (475, 190),
                35 + 8 * slide_index,
                (30 + 20 * slide_index, 70, 150),
                5,
            )

            if moving_occluder:
                x = 10 + int((timestamp * 57) % 140)
                cv2.rectangle(frame, (x, 215), (x + 70, 350), (45, 45, 45), -1)

            if luminance_flicker_at is not None and abs(timestamp - luminance_flicker_at) < 0.11:
                frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=38)
            writer.write(frame)
    finally:
        writer.release()


def make_vfr_slide_video(path: Path, transitions: list[float]) -> None:
    """Write a sparse VFR lecture fixture with known presentation-time changes."""
    if len(transitions) != 2:
        raise ValueError("VFR oracle expects exactly two transitions")

    frame_paths = []
    palette = [(245, 245, 245), (210, 240, 250), (245, 215, 235)]
    for index in range(3):
        frame = np.full((360, 640, 3), palette[index], dtype=np.uint8)
        cv2.rectangle(frame, (24, 20), (616, 334), (80, 80, 80), 2)
        cv2.putText(
            frame,
            f"VFR SLIDE {index + 1}",
            (48, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (15, 15, 15),
            3,
            cv2.LINE_AA,
        )
        cv2.rectangle(
            frame,
            (80 + index * 120, 145),
            (360 + index * 90, 290),
            (40 + index * 60, 70, 180 - index * 50),
            -1,
        )
        frame_path = path.parent / f"vfr-{index}.png"
        if not cv2.imwrite(str(frame_path), frame):
            raise RuntimeError(f"Could not write VFR fixture frame: {frame_path}")
        frame_paths.append(frame_path)

    timestamps = [
        0.0,
        0.2,
        0.52,
        1.0,
        transitions[0],
        2.0,
        2.52,
        3.0,
        3.52,
        4.0,
        4.52,
        5.0,
        transitions[1],
        6.0,
        6.52,
    ]
    concat = path.parent / "vfr-frames.txt"
    lines = []
    selected_frames = [
        frame_paths[sum(timestamp >= transition for transition in transitions)]
        for timestamp in timestamps
    ]
    durations = [
        next_timestamp - timestamp
        for timestamp, next_timestamp in zip(timestamps, timestamps[1:], strict=False)
    ]
    durations.append(0.5)
    for frame_path, duration in zip(selected_frames, durations, strict=True):
        lines.extend([f"file '{frame_path.name}'", f"duration {duration}"])
    lines.append(f"file '{selected_frames[-1].name}'")
    concat.write_text("\n".join(lines) + "\n", encoding="utf-8")
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat),
            "-fps_mode",
            "vfr",
            "-c:v",
            "ffv1",
            "-y",
            str(path),
        ],
        check=True,
    )
