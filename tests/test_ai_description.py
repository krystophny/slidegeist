"""Behavioral tests for slide-description text handling and configuration."""

import pytest

from slidegeist.ai_description import (
    LlamaCppSlideDescriber,
    clean_text,
    get_user_prompt,
)


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
    assert "stop immediately after the word NON-SLIDE" in prompt
    assert "If the frame type is SLIDE, continue" in prompt
    assert "Aim for at most 200 words" in prompt
    assert "do not repeat TEXT CONTENT" in prompt


def test_description_token_budget_is_configurable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SLIDEGEIST_LLAMACPP_MODEL", "oracle")
    monkeypatch.setenv("SLIDEGEIST_LLAMACPP_MAX_TOKENS", "640")

    assert LlamaCppSlideDescriber().max_new_tokens == 640


def test_description_token_budget_rejects_too_small_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SLIDEGEIST_LLAMACPP_MODEL", "oracle")
    monkeypatch.setenv("SLIDEGEIST_LLAMACPP_MAX_TOKENS", "128")

    with pytest.raises(ValueError, match="must be at least 256"):
        LlamaCppSlideDescriber()
