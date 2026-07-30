#!/usr/bin/env python3
"""Reproducible synthetic benchmark for slide transition detection."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slidegeist.transition_detector import analyze_slide_transitions
from tests.synthetic_video import make_slide_video


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, help="Optional JSON report destination")
    args = parser.parse_args()
    oracle = [2.7, 8.9, 11.0, 14.6]

    with tempfile.TemporaryDirectory(prefix="slidegeist-benchmark-") as temp_dir:
        video = Path(temp_dir) / "oracle.avi"
        make_slide_video(video, oracle, duration=17.0)
        start = time.perf_counter()
        analysis = analyze_slide_transitions(
            video,
            start_offset=0.0,
            min_scene_len=0.8,
        )
        elapsed = time.perf_counter() - start

    matched = sum(
        any(abs(detected - expected) <= 0.65 for detected in analysis.timestamps)
        for expected in oracle
    )
    false_positives = sum(
        not any(abs(detected - expected) <= 0.65 for expected in oracle)
        for detected in analysis.timestamps
    )
    precision = matched / max(matched + false_positives, 1)
    recall = matched / len(oracle)
    report = {
        "fixture": "synthetic variable cadence + moving occluder + luminance flicker",
        "oracle_seconds": oracle,
        "detected_seconds": analysis.timestamps,
        "tolerance_seconds": 0.65,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "video_seconds": 17.0,
        "elapsed_seconds": elapsed,
        "realtime_factor": elapsed / 17.0,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.json:
        args.json.write_text(rendered + "\n", encoding="utf-8")
    return 0 if precision == 1.0 and recall == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
