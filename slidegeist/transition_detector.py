"""Robust slide-transition detection from complementary visual evidence.

The detector deliberately does not target a slide count or assume a cadence.
It samples the video, establishes robust per-video baselines, fuses structural,
colour, edge, perceptual, and spatial-coverage evidence, and keeps local peaks.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameEvidence:
    """Visual change evidence between two sampled frames."""

    timestamp: float
    ssim_loss: float
    histogram: float
    edge_change: float
    phash: float
    coverage: float
    luminance: float
    score: float = 0.0
    candidate: bool = False


@dataclass(frozen=True)
class TransitionAnalysis:
    """Detected transitions plus auditable feature and threshold data."""

    timestamps: list[float]
    sample_interval: float
    thresholds: dict[str, float]
    evidence: list[FrameEvidence]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "method": "robust-visual-ensemble-v1",
            "timestamps": self.timestamps,
            "sample_interval": self.sample_interval,
            "thresholds": self.thresholds,
            "evidence": [asdict(item) for item in self.evidence],
        }


def _resize(frame: np.ndarray, max_width: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / width
    return cv2.resize(
        frame,
        (max_width, max(2, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _ssim_loss(previous: np.ndarray, current: np.ndarray) -> float:
    """Return mean local SSIM dissimilarity in [0, 1]."""
    x = previous.astype(np.float32)
    y = current.astype(np.float32)
    mu_x = cv2.GaussianBlur(x, (7, 7), 1.5)
    mu_y = cv2.GaussianBlur(y, (7, 7), 1.5)
    sigma_x = cv2.GaussianBlur(x * x, (7, 7), 1.5) - mu_x * mu_x
    sigma_y = cv2.GaussianBlur(y * y, (7, 7), 1.5) - mu_y * mu_y
    sigma_xy = cv2.GaussianBlur(x * y, (7, 7), 1.5) - mu_x * mu_y
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    similarity = np.divide(
        numerator,
        denominator,
        out=np.ones_like(numerator),
        where=denominator > 1e-6,
    )
    return float(np.clip(1.0 - np.mean(similarity), 0.0, 1.0))


def _histogram_distance(previous: np.ndarray, current: np.ndarray) -> float:
    prev_hsv = cv2.cvtColor(previous, cv2.COLOR_BGR2HSV)
    curr_hsv = cv2.cvtColor(current, cv2.COLOR_BGR2HSV)
    prev_hist = cv2.calcHist([prev_hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    curr_hist = cv2.calcHist([curr_hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    cv2.normalize(prev_hist, prev_hist, alpha=1.0, norm_type=cv2.NORM_L1)
    cv2.normalize(curr_hist, curr_hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return float(cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_BHATTACHARYYA))


def _edge_change(previous: np.ndarray, current: np.ndarray) -> float:
    prev_edges = cv2.Canny(previous, 60, 180)
    curr_edges = cv2.Canny(current, 60, 180)
    kernel = np.ones((3, 3), np.uint8)
    prev_edges = cv2.dilate(prev_edges, kernel)
    curr_edges = cv2.dilate(curr_edges, kernel)
    union = np.count_nonzero((prev_edges > 0) | (curr_edges > 0))
    if union == 0:
        return 0.0
    changed = np.count_nonzero((prev_edges > 0) ^ (curr_edges > 0))
    return float(changed / union)


def _phash(gray: np.ndarray) -> np.ndarray:
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    coefficients = cv2.dct(small.astype(np.float32))[:8, :8]
    usable = coefficients.flatten()[1:]
    return usable > np.median(usable)


def _coverage_and_luminance(previous: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    absolute = cv2.absdiff(previous, current).astype(np.float32) / 255.0
    height, width = absolute.shape
    tile_h = max(1, height // 6)
    tile_w = max(1, width // 8)
    tile_changes: list[float] = []
    for y in range(0, height, tile_h):
        for x in range(0, width, tile_w):
            tile = absolute[y : min(y + tile_h, height), x : min(x + tile_w, width)]
            if tile.size:
                tile_changes.append(float(np.mean(tile)))
    coverage = float(np.mean(np.asarray(tile_changes) >= 0.045)) if tile_changes else 0.0
    return coverage, float(np.mean(absolute))


def _raw_evidence(previous: np.ndarray, current: np.ndarray, timestamp: float) -> FrameEvidence:
    prev_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    coverage, luminance = _coverage_and_luminance(prev_gray, curr_gray)
    phash_distance = float(np.mean(_phash(prev_gray) != _phash(curr_gray)))
    return FrameEvidence(
        timestamp=timestamp,
        ssim_loss=_ssim_loss(prev_gray, curr_gray),
        histogram=_histogram_distance(previous, current),
        edge_change=_edge_change(prev_gray, curr_gray),
        phash=phash_distance,
        coverage=coverage,
        luminance=luminance,
    )


_FLOORS = {
    "ssim_loss": 0.085,
    "histogram": 0.10,
    "edge_change": 0.22,
    "phash": 0.075,
    "coverage": 0.14,
    "luminance": 0.035,
}


def _robust_threshold(values: np.ndarray, floor: float) -> float:
    if values.size == 0:
        return floor
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return max(floor, median + 6.0 * 1.4826 * mad)


def _score_evidence(
    evidence: list[FrameEvidence],
    threshold_bias: float,
) -> tuple[list[FrameEvidence], dict[str, float]]:
    thresholds = {
        name: _robust_threshold(
            np.asarray([getattr(item, name) for item in evidence], dtype=np.float64),
            floor,
        )
        for name, floor in _FLOORS.items()
    }
    scored: list[FrameEvidence] = []
    for item in evidence:
        ratios = {
            name: min(getattr(item, name) / threshold, 3.0)
            for name, threshold in thresholds.items()
        }
        score = (
            0.28 * ratios["ssim_loss"]
            + 0.20 * ratios["histogram"]
            + 0.18 * ratios["edge_change"]
            + 0.18 * ratios["phash"]
            + 0.12 * ratios["coverage"]
            + 0.04 * ratios["luminance"]
        )
        structural_votes = sum(
            ratios[name] >= 1.0 for name in ("ssim_loss", "edge_change", "phash")
        )
        global_support = ratios["coverage"] >= 1.0 or ratios["histogram"] >= 1.0
        structural_support = structural_votes >= 2 or (
            structural_votes >= 1 and score >= 1.25 + threshold_bias
        )
        candidate = global_support and structural_support and score >= 1.0 + threshold_bias
        scored.append(
            FrameEvidence(
                timestamp=item.timestamp,
                ssim_loss=item.ssim_loss,
                histogram=item.histogram,
                edge_change=item.edge_change,
                phash=item.phash,
                coverage=item.coverage,
                luminance=item.luminance,
                score=score,
                candidate=candidate,
            )
        )
    thresholds["score"] = 1.0 + threshold_bias
    return scored, thresholds


def _select_peaks(evidence: list[FrameEvidence], min_scene_len: float) -> list[float]:
    candidates = [item for item in evidence if item.candidate]
    if not candidates:
        return []

    groups: list[list[FrameEvidence]] = [[candidates[0]]]
    for item in candidates[1:]:
        if item.timestamp - groups[-1][-1].timestamp <= max(min_scene_len, 0.01):
            groups[-1].append(item)
        else:
            groups.append([item])
    return [max(group, key=lambda item: item.score).timestamp for group in groups]


def analyze_slide_transitions(
    video_path: Path,
    *,
    threshold: float = 0.025,
    start_offset: float = 3.0,
    min_scene_len: float = 0.75,
    sample_interval: float = 0.5,
    max_width: int = 480,
) -> TransitionAnalysis:
    """Analyze a video for slide transitions without a cadence assumption.

    ``threshold`` is retained for CLI compatibility and only biases the
    evidence threshold: values below 0.025 are more sensitive, values above
    are less sensitive. It never determines or caps the number of slides.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        capture.release()
        raise RuntimeError(f"Video reports invalid FPS: {fps}")
    frame_step = max(1, round(sample_interval * fps))
    sample_interval = frame_step / fps
    start_frame = max(0, round(start_offset * fps))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    evidence: list[FrameEvidence] = []
    previous: np.ndarray | None = None

    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_index = start_frame
        next_sample = start_frame
        while frame_index < total_frames and capture.grab():
            if frame_index >= next_sample:
                ok, frame = capture.retrieve()
                if not ok:
                    break
                current = _resize(frame, max_width)
                if previous is not None:
                    evidence.append(_raw_evidence(previous, current, frame_index / fps))
                previous = current
                next_sample += frame_step
            frame_index += 1
    finally:
        capture.release()

    threshold_bias = float(np.clip((threshold - 0.025) * 8.0, -0.18, 0.30))
    scored, thresholds = _score_evidence(evidence, threshold_bias)
    timestamps = _select_peaks(scored, min_scene_len)
    logger.info(
        "Robust visual ensemble found %d transitions from %d samples (no cadence prior)",
        len(timestamps),
        len(scored),
    )
    return TransitionAnalysis(timestamps, sample_interval, thresholds, scored)


def detect_slide_transitions(
    video_path: Path,
    *,
    threshold: float = 0.025,
    start_offset: float = 3.0,
    min_scene_len: float = 0.75,
) -> list[float]:
    """Return slide-transition timestamps from robust visual evidence."""
    return analyze_slide_transitions(
        video_path,
        threshold=threshold,
        start_offset=start_offset,
        min_scene_len=min_scene_len,
    ).timestamps
