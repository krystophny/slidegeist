"""Behavioral resume-stage tests."""

import json
from pathlib import Path

import pytest

from slidegeist.pipeline import (
    detect_completed_stages,
    load_transcript_checkpoint,
    process_video,
)


def _write_slide(output: Path, number: int) -> None:
    slides = output / "slides"
    slides.mkdir(parents=True, exist_ok=True)
    (slides / f"slide_{number:03d}.jpg").write_bytes(b"jpeg fixture")


def _section(number: int, description: str = "") -> str:
    description_section = (
        f"### AI Description (for reconstruction)\n\n{description}\n\n"
        if description
        else ""
    )
    return (
        f'<a name="slide_{number:03d}"></a>\n\n'
        f"## Slide {number}\n\n"
        f"{description_section}---\n"
    )


def test_partial_ai_descriptions_remain_resumable(tmp_path: Path) -> None:
    _write_slide(tmp_path, 1)
    _write_slide(tmp_path, 2)
    (tmp_path / "slides.md").write_text(
        _section(1, "0. FRAME TYPE\nSLIDE\n\nComplete first description")
        + _section(2),
        encoding="utf-8",
    )

    assert not detect_completed_stages(tmp_path)["ai_description"]


def test_all_ai_descriptions_mark_stage_complete(tmp_path: Path) -> None:
    _write_slide(tmp_path, 1)
    _write_slide(tmp_path, 2)
    (tmp_path / "slides.md").write_text(
        _section(1, "0. FRAME TYPE\nSLIDE\n\nComplete first description")
        + _section(2, "0. FRAME TYPE\nSLIDE\n\nComplete second description"),
        encoding="utf-8",
    )

    assert detect_completed_stages(tmp_path)["ai_description"]


def test_legacy_description_without_frame_type_remains_resumable(
    tmp_path: Path,
) -> None:
    _write_slide(tmp_path, 1)
    (tmp_path / "slides.md").write_text(
        _section(1, "1. TITLE\nLegacy ambiguous response"),
        encoding="utf-8",
    )

    assert not detect_completed_stages(tmp_path)["ai_description"]


def test_filter_plan_with_unexported_rejected_section_remains_resumable(
    tmp_path: Path,
) -> None:
    _write_slide(tmp_path, 1)
    rejected = tmp_path / "slides" / "slide_002.jpg.non-slide"
    rejected.write_bytes(b"known rejected frame")
    (tmp_path / "slides.md").write_text(
        _section(1, "0. FRAME TYPE\nSLIDE\n\nAccepted")
        + _section(2, "0. FRAME TYPE\nNON-SLIDE\n\nRejected"),
        encoding="utf-8",
    )

    assert not detect_completed_stages(tmp_path)["ai_description"]


def test_split_export_with_classified_description_is_complete(tmp_path: Path) -> None:
    _write_slide(tmp_path, 1)
    (tmp_path / "index.md").write_text("# Lecture Slides\n", encoding="utf-8")
    (tmp_path / "slide_001.md").write_text(
        "---\nid: slide_001\nindex: 1\ntime_start: 0.0\ntime_end: 10.0\n---\n\n"
        "# Slide 1\n\n"
        "## AI Description (for reconstruction)\n\n"
        "0. FRAME TYPE\nSLIDE\n\n1. TITLE\nKnown split fixture\n",
        encoding="utf-8",
    )

    assert detect_completed_stages(tmp_path)["ai_description"]


def test_resume_loads_timed_transcript_context(tmp_path: Path) -> None:
    """A durable checkpoint preserves the speech window used after restart."""
    checkpoint = tmp_path / "transcript.json"
    checkpoint.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "start": 12.5,
                        "end": 15.0,
                        "text": "  The magnetic moment is conserved. ",
                        "words": [
                            {
                                "word": "moment",
                                "start": 12.9,
                                "end": 13.2,
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert load_transcript_checkpoint(checkpoint) == [
        {
            "start": 12.5,
            "end": 15.0,
            "text": "The magnetic moment is conserved.",
            "words": [{"word": "moment", "start": 12.9, "end": 13.2}],
        }
    ]


def test_unretried_failed_stage_returns_a_processing_error(tmp_path: Path) -> None:
    video = tmp_path / "known.mp4"
    video.write_bytes(b"known video placeholder")
    _write_slide(tmp_path, 1)
    (tmp_path / "transcript.json").write_text(
        json.dumps(
            {"segments": [{"start": 0.0, "end": 1.0, "text": "Known speech"}]}
        ),
        encoding="utf-8",
    )
    (tmp_path / "slides.md").write_text(
        _section(1) + "\n### OCR Text\nKnown OCR\n",
        encoding="utf-8",
    )
    (tmp_path / ".ai_description_failed").write_text(
        "independent service failure", encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="failed stages: ai_description"):
        process_video(video, tmp_path)
