"""Behavioral tests for slide-description text handling and configuration."""

from pathlib import Path

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
    prompt = get_user_prompt("", "Topic . . . . . . 12")

    assert 'Write exactly "SLIDE"' in prompt
    assert 'Write exactly "NON-SLIDE"' in prompt
    assert "file browser" in prompt
    assert "substantially obscured by operating-system UI" in prompt
    assert "stop immediately after the word NON-SLIDE" in prompt
    assert "If the frame type is SLIDE, continue" in prompt
    assert "Aim for at most 200 words" in prompt
    assert "do not repeat TEXT CONTENT" in prompt
    assert 'once as "[leader]"' in prompt
    assert "Topic [leader] 12" in prompt
    assert ". . ." not in prompt


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


def test_incomplete_visual_description_gets_one_text_only_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "contents.jpg"
    image.write_bytes(b"image")
    calls: list[Path | None] = []
    complete = (
        "0. FRAME TYPE\nSLIDE\n\n1. TITLE\nContents\n\n"
        "2. TEXT CONTENT\nTopic [leader] 12\n\n3. FORMULAS\nNone\n\n"
        "4. VISUAL ELEMENTS\nNone\n\n5. LAYOUT\nOne-column list"
    )

    def fake_complete(_: str, *, image_path: Path | None = None, **__: object) -> str:
        calls.append(image_path)
        if image_path is not None:
            return "0. FRAME TYPE\nSLIDE\n\n1. TITLE\nContents\n\n2. TEXT CONTENT\n. . ."
        return complete

    monkeypatch.setattr("slidegeist.ai_description.llama_cpp_complete", fake_complete)

    result = LlamaCppSlideDescriber(max_new_tokens=1024).describe(
        image, "lecture context", "Topic . . . . . . 12"
    )

    assert result == complete
    assert calls == [image, None]
