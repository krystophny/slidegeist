"""Behavioral tests for slide-description text handling."""

from slidegeist.ai_description import clean_text


def test_clean_text_preserves_reconstruction_sections() -> None:
    raw = "1. TITLE\n\nPlasma   Physics\n\n\n\n2. TEXT CONTENT\n• density"

    cleaned = clean_text(raw)

    assert cleaned.splitlines() == [
        "1. TITLE",
        "",
        "Plasma Physics",
        "",
        "2. TEXT CONTENT",
        "• density",
    ]
