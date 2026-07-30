"""Behavioral tests for slide-description text handling."""

from slidegeist.ai_description import clean_text, get_user_prompt


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


def test_prompt_defines_non_instructional_ui_as_non_slide() -> None:
    prompt = get_user_prompt("", "")

    assert 'Write exactly "SLIDE"' in prompt
    assert 'Write exactly "NON-SLIDE"' in prompt
    assert "file browser" in prompt
    assert "substantially obscured by operating-system UI" in prompt
