"""Behavioral oracles for rejecting non-instructional visual states."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from slidegeist.frame_filter import (
    filter_non_slide_states,
    frame_type,
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
    assert payload["state_filter_video_end"] == 30.0


def test_frame_type_parser_requires_an_unambiguous_structured_decision() -> None:
    assert is_non_slide_description(_description("NON-SLIDE"))
    assert frame_type(_description("SLIDE")) == "SLIDE"
    assert frame_type("SLIDE\n\n1. TITLE\n8.1 Fusion in a Test Tube?") == "SLIDE"
    assert frame_type("NON_SLIDE\n\n1. TITLE\nDesktop") == "NON-SLIDE"
    assert frame_type("1. TITLE\nLegacy accepted slide") is None
    assert frame_type(_description("SLIDE") + "\n" + _description("NON-SLIDE")) is None
    assert frame_type("SLIDE\n\n" + _description("NON-SLIDE")) is None


def test_filter_rejects_a_description_without_frame_type(tmp_path: Path) -> None:
    slides = tmp_path / "slides"
    slides.mkdir()
    image = slides / "slide_001.jpg"
    image.write_bytes(b"known instructional frame")
    diagnostics = tmp_path / "transition_detection.json"
    diagnostics.write_text('{"timestamps": []}', encoding="utf-8")

    try:
        filter_non_slide_states(
            [(1, 0.0, 10.0, image)],
            {"slide_001": "1. TITLE\nLegacy ambiguous response"},
            diagnostics,
        )
    except RuntimeError as exc:
        assert "lack an unambiguous frame type" in str(exc)
    else:
        raise AssertionError("ambiguous description was silently accepted")


def test_filter_recovers_after_only_some_planned_renames(tmp_path: Path) -> None:
    slides = tmp_path / "slides"
    slides.mkdir()
    images = [slides / f"slide_{index:03d}.jpg" for index in range(1, 4)]
    for index, image in enumerate(images, 1):
        image.write_bytes(f"independent frame {index}".encode())
    rejected_states = [
        {
            "slide_id": image.stem,
            "start": float((index - 1) * 10),
            "end": float(index * 10),
            "image_sha256": sha256(image.read_bytes()).hexdigest(),
            "classification": _description("NON-SLIDE"),
        }
        for index, image in enumerate(images[1:], 2)
    ]
    diagnostics = tmp_path / "transition_detection.json"
    diagnostics.write_text(
        json.dumps(
            {
                "timestamps": [],
                "raw_timestamps": [10.0, 20.0],
                "rejected_states": rejected_states,
                "state_filter": "multimodal-instructional-content-v1",
                "state_filter_video_end": 30.0,
            }
        ),
        encoding="utf-8",
    )
    images[1].rename(images[1].with_suffix(".jpg.non-slide"))

    filtered = filter_non_slide_states(
        [
            (1, 0.0, 10.0, images[0]),
            (3, 20.0, 30.0, images[2]),
        ],
        {
            "slide_001": _description("SLIDE"),
            "slide_003": _description("NON-SLIDE"),
        },
        diagnostics,
    )

    assert filtered == [(1, 0.0, 30.0, images[0])]
    assert images[2].with_suffix(".jpg.non-slide").exists()
