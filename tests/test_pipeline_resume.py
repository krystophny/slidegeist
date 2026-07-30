"""Behavioral resume-stage tests."""

from pathlib import Path

from slidegeist.pipeline import detect_completed_stages


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
        _section(1, "Complete first description") + _section(2),
        encoding="utf-8",
    )

    assert not detect_completed_stages(tmp_path)["ai_description"]


def test_all_ai_descriptions_mark_stage_complete(tmp_path: Path) -> None:
    _write_slide(tmp_path, 1)
    _write_slide(tmp_path, 2)
    (tmp_path / "slides.md").write_text(
        _section(1, "Complete first description")
        + _section(2, "Complete second description"),
        encoding="utf-8",
    )

    assert detect_completed_stages(tmp_path)["ai_description"]
