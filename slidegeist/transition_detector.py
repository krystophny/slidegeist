"""Robust slide-transition detection from complementary visual evidence.

The detector deliberately does not target a slide count or assume a cadence.
It samples the video, establishes robust per-video baselines, fuses structural,
colour, edge, perceptual, and spatial-coverage evidence, and keeps local peaks.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace
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
    return np.asarray(usable > np.median(usable), dtype=np.bool_)


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


def _remove_unstable_terminal_peak(
    timestamps: list[float],
    *,
    last_sample_timestamp: float,
    min_scene_len: float,
    sample_interval: float,
) -> list[float]:
    """Reject a terminal flash/fade with no sampled evidence of a stable scene."""
    stability_window = max(0.01, min_scene_len - sample_interval)
    return [
        timestamp
        for timestamp in timestamps
        if last_sample_timestamp - timestamp >= stability_window
    ]


def _read_exact(stream: Any, size: int) -> bytes:
    """Read one fixed-size raw frame from a pipe."""
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            break
        chunks.extend(chunk)
    return bytes(chunks)


def _sampled_frames(
    video_path: Path,
    *,
    start_offset: float,
    sample_interval: float,
    max_width: int,
    source_width: int,
    source_height: int,
    timestamps: list[float],
) -> Iterator[np.ndarray]:
    """Yield resized frames directly from FFmpeg at the requested sample rate.

    Sampling in FFmpeg avoids transferring every full-resolution decoded frame
    through Python. Selection uses source presentation times, so variable-frame-
    rate inputs remain a regular temporal evidence grid. This is not a cadence
    prior.
    """
    output_width = min(max_width, source_width)
    output_height = max(
        2,
        round((source_height * output_width / source_width) / 2) * 2,
    )
    frame_bytes = output_width * output_height * 3
    start = format(start_offset, ".12g")
    interval = format(sample_interval, ".12g")
    select = f"select=gte(t\\,{start}+selected_n*{interval})"
    stats_file = tempfile.NamedTemporaryFile(prefix="slidegeist-pts-", delete=False)
    stats_path = Path(stats_file.name)
    stats_file.close()
    try:
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vf",
            f"{select},scale={output_width}:{output_height}:flags=area",
            "-an",
            "-fps_mode",
            "vfr",
            "-stats_enc_pre:v:0",
            str(stats_path),
            "-stats_enc_pre_fmt:v:0",
            "{ti}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdout is not None
        assert process.stderr is not None
        completed_read = False
        try:
            while True:
                raw = _read_exact(process.stdout, frame_bytes)
                if not raw:
                    completed_read = True
                    break
                if len(raw) != frame_bytes:
                    raise RuntimeError(
                        f"FFmpeg returned a partial sampled frame ({len(raw)}/{frame_bytes} bytes)"
                    )
                yield np.frombuffer(raw, dtype=np.uint8).reshape(
                    output_height,
                    output_width,
                    3,
                )
        finally:
            if not completed_read and process.poll() is None:
                process.terminate()
            process.stdout.close()
            stderr = process.stderr.read().decode("utf-8", errors="replace").strip()
            return_code = process.wait()
            process.stderr.close()
            if return_code and completed_read:
                raise RuntimeError(f"FFmpeg sampled-frame decode failed: {stderr}")
        timestamps.extend(
            float(line)
            for line in stats_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    finally:
        stats_path.unlink(missing_ok=True)


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
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if fps <= 0:
        raise RuntimeError(f"Video reports invalid nominal FPS: {fps}")
    if source_width <= 0 or source_height <= 0:
        raise RuntimeError(f"Video reports invalid dimensions: {source_width}x{source_height}")
    sample_interval = max(1, round(sample_interval * fps)) / fps

    evidence: list[FrameEvidence] = []
    previous: np.ndarray | None = None
    timestamps: list[float] = []

    for current in _sampled_frames(
        video_path,
        start_offset=start_offset,
        sample_interval=sample_interval,
        max_width=max_width,
        source_width=source_width,
        source_height=source_height,
        timestamps=timestamps,
    ):
        if previous is not None:
            evidence.append(_raw_evidence(previous, current, 0.0))
        previous = current

    if len(timestamps) != len(evidence) + bool(previous is not None):
        raise RuntimeError(
            "FFmpeg timestamp/frame count mismatch: "
            f"{len(timestamps)} timestamps for {len(evidence) + bool(previous is not None)} frames"
        )
    evidence = [
        replace(item, timestamp=timestamp)
        for item, timestamp in zip(evidence, timestamps[1:], strict=True)
    ]

    threshold_bias = float(np.clip((threshold - 0.025) * 8.0, -0.18, 0.30))
    scored, thresholds = _score_evidence(evidence, threshold_bias)
    timestamps = _select_peaks(scored, min_scene_len)
    selected_count = len(timestamps)
    if evidence:
        timestamps = _remove_unstable_terminal_peak(
            timestamps,
            last_sample_timestamp=evidence[-1].timestamp,
            min_scene_len=min_scene_len,
            sample_interval=sample_interval,
        )
    thresholds["terminal_stability"] = max(0.01, min_scene_len - sample_interval)
    discarded_terminal = selected_count - len(timestamps)
    logger.info(
        "Robust visual ensemble found %d transitions from %d samples "
        "(%d unstable terminal flash/fade discarded; no cadence prior)",
        len(timestamps),
        len(scored),
        discarded_terminal,
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
