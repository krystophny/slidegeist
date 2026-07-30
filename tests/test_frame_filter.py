"""Behavioral oracles for rejecting non-instructional visual states."""

from __future__ import annotations

import json
from pathlib import Path

from slidegeist.frame_filter import (
    filter_non_slide_states,
    is_non_slide_description,
)


def _description(frame_type: str) -> str:
    return f"0. FRAME TYPE\n{frame_type}\n\n1. TITLE\nKnown fixture"


def test_non_slide_state_is_removed_and_neighboring_intervals_merge(
    tmp_path: Path,
) -> None:
    slides = tmp_path / "slides"
    slides.mkdir()
    images = [slides / f"slide_{index:03d}.jpg" for index in range(1, 4)]
    for index, image in enumerate(images, 1):
        image.write_bytes(f"independent frame {index}".encode())
    diagnostics = tmp_path / "transition_detection.json"
    diagnostics.write_text(
        json.dumps({"timestamps": [10.0, 20.0], "evidence": []}),
        encoding="utf-8",
    )
    metadata = [
        (1, 0.0, 10.0, images[0]),
        (2, 10.0, 20.0, images[1]),
        (3, 20.0, 30.0, images[2]),
    ]
    descriptions = {
        "slide_001": _description("SLIDE"),
        "slide_002": _description("NON-SLIDE"),
        "slide_003": _description("SLIDE"),
    }

    filtered = filter_non_slide_states(metadata, descriptions, diagnostics)

    assert filtered == [
        (1, 0.0, 20.0, images[0]),
        (3, 20.0, 30.0, images[2]),
    ]
    assert not images[1].exists()
    assert images[1].with_suffix(".jpg.non-slide").exists()
    payload = json.loads(diagnostics.read_text())
    assert payload["raw_timestamps"] == [10.0, 20.0]
    assert payload["timestamps"] == [20.0]
    assert payload["rejected_states"][0]["slide_id"] == "slide_002"
    assert payload["rejected_states"][0]["image_sha256"]


def test_frame_type_parser_does_not_reject_legacy_descriptions() -> None:
    assert is_non_slide_description(_description("NON-SLIDE"))
    assert not is_non_slide_description(_description("SLIDE"))
    assert not is_non_slide_description("1. TITLE\nLegacy accepted slide")
