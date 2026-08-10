"""Tests for speaker-turn assignment and the DiariZen subprocess boundary."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from slidegeist import diarize
from slidegeist.diarize import (
    DiariZenDiarizer,
    assign_speakers,
    build_diarizer,
    canonicalize_speaker_ids,
)


def _segment(start, end, text, words=None):
    return {"start": start, "end": end, "text": text, "words": words or []}


def test_words_get_dominant_overlap_speaker() -> None:
    segments = [
        _segment(
            0.0,
            4.0,
            "hello there friend",
            [
                {"word": "hello", "start": 0.0, "end": 1.0},
                {"word": "there", "start": 1.0, "end": 2.0},
                {"word": "friend", "start": 3.0, "end": 4.0},
            ],
        )
    ]
    turns = [
        {"start": 0.0, "end": 2.5, "speaker": "A"},
        {"start": 2.5, "end": 4.0, "speaker": "B"},
    ]

    assign_speakers(segments, turns)

    assert [word["speaker"] for word in segments[0]["words"]] == ["A", "A", "B"]


def test_segment_speaker_follows_word_time_not_word_count() -> None:
    """One long term must outweigh a run of short function words."""
    segments = [
        _segment(
            0.0,
            10.0,
            "and so the magnetohydrodynamics",
            [
                {"word": "and", "start": 0.0, "end": 0.3},
                {"word": "so", "start": 0.3, "end": 0.6},
                {"word": "the", "start": 0.6, "end": 0.9},
                {"word": "magnetohydrodynamics", "start": 2.0, "end": 10.0},
            ],
        )
    ]
    turns = [
        {"start": 0.0, "end": 1.0, "speaker": "SHORT"},
        {"start": 1.0, "end": 10.0, "speaker": "LONG"},
    ]

    assign_speakers(segments, turns)

    assert segments[0]["speaker"] == "LONG"


def test_wordless_segment_uses_segment_overlap() -> None:
    """The whisper path can legitimately have no word timing at all."""
    segments = [_segment(0.0, 4.0, "no words here")]
    turns = [
        {"start": 0.0, "end": 1.0, "speaker": "A"},
        {"start": 1.0, "end": 4.0, "speaker": "B"},
    ]

    assign_speakers(segments, turns)

    assert segments[0]["speaker"] == "B"


def test_word_outside_all_turns_keeps_speaker_unset() -> None:
    segments = [
        _segment(
            0.0,
            30.0,
            "far away",
            [{"word": "far", "start": 20.0, "end": 21.0}],
        )
    ]
    turns = [{"start": 0.0, "end": 1.0, "speaker": "A"}]

    assign_speakers(segments, turns)

    assert "speaker" not in segments[0]["words"][0]


def test_overlapping_turns_record_secondary_speakers() -> None:
    segments = [_segment(0.0, 4.0, "crosstalk")]
    turns = [
        {"start": 0.0, "end": 3.0, "speaker": "A"},
        {"start": 2.0, "end": 4.0, "speaker": "B"},
    ]

    assign_speakers(segments, turns)

    shares = segments[0]["speakers"]
    assert [share["speaker"] for share in shares] == ["A", "B"]
    assert shares[0]["seconds"] == pytest.approx(3.0)
    assert shares[1]["seconds"] == pytest.approx(2.0)


def test_single_speaker_segment_records_no_shares() -> None:
    """Single-speaker material must stay byte-identical to schema v1 plus one key."""
    segments = [_segment(0.0, 4.0, "one voice")]
    turns = [{"start": 0.0, "end": 4.0, "speaker": "A"}]

    assign_speakers(segments, turns)

    assert "speakers" not in segments[0]


def test_speaker_ids_renumbered_in_first_appearance_order() -> None:
    segments = [
        _segment(0.0, 1.0, "second speaker first"),
        _segment(1.0, 2.0, "first speaker second"),
    ]
    segments[0]["speaker"] = "7"
    segments[1]["speaker"] = "3"
    turns = [
        {"start": 0.0, "end": 1.0, "speaker": "7"},
        {"start": 1.0, "end": 2.0, "speaker": "3"},
    ]

    names = canonicalize_speaker_ids(segments, turns)

    assert segments[0]["speaker"] == "SPEAKER_00"
    assert segments[1]["speaker"] == "SPEAKER_01"
    assert names == ["SPEAKER_00", "SPEAKER_01"]
    assert [turn["speaker"] for turn in turns] == ["SPEAKER_00", "SPEAKER_01"]


def test_worker_argv_targets_configured_interpreter(monkeypatch, tmp_path) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        payload = {
            "schema": diarize.DIARIZATION_SCHEMA,
            "model": "m",
            "audio_duration": 1.0,
            "turns": [{"start": 0.0, "end": 1.0, "speaker": "0"}],
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    diarizer = DiariZenDiarizer(interpreter="/usr/bin/python3.11", model="m", device="cpu")
    turns = diarizer.diarize(tmp_path / "audio.wav")

    assert turns == [{"start": 0.0, "end": 1.0, "speaker": "0"}]
    command = captured["command"]
    assert command[0] == "/usr/bin/python3.11"
    assert "-m" not in command, "the worker runs as a file; the venv has no slidegeist"
    assert command[2].endswith("_diarizen_worker.py")


def test_worker_rejects_unknown_schema(monkeypatch, tmp_path) -> None:
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, json.dumps({"schema": "nope"}), "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    diarizer = DiariZenDiarizer(interpreter="/usr/bin/python3.11", model="m")

    with pytest.raises(RuntimeError, match="Unexpected diarization schema"):
        diarizer.diarize(tmp_path / "audio.wav")


def test_nonzero_exit_reports_stderr_tail(monkeypatch, tmp_path) -> None:
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, "", "boom: model missing")

    monkeypatch.setattr(subprocess, "run", fake_run)
    diarizer = DiariZenDiarizer(interpreter="/usr/bin/python3.11", model="m")

    with pytest.raises(RuntimeError, match="boom: model missing"):
        diarizer.diarize(tmp_path / "audio.wav")


def test_cached_diarization_skips_subprocess(monkeypatch, tmp_path) -> None:
    cache = tmp_path / "diarization.json"
    cache.write_text(
        json.dumps(
            {
                "schema": diarize.DIARIZATION_SCHEMA,
                "model": "m",
                "audio_duration": 1.0,
                "turns": [{"start": 0.0, "end": 1.0, "speaker": "0"}],
            }
        ),
        encoding="utf-8",
    )

    def explode(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("subprocess must not be spawned when a cache is valid")

    monkeypatch.setattr(subprocess, "run", explode)
    diarizer = DiariZenDiarizer(interpreter="/usr/bin/python3.11", model="m")

    assert diarizer.diarize(tmp_path / "audio.wav", cache_path=cache)


def test_build_diarizer_modes(monkeypatch) -> None:
    monkeypatch.delenv("SLIDEGEIST_DIARIZEN_PYTHON", raising=False)

    assert build_diarizer("off") is None
    assert build_diarizer("provider") is None
    # auto degrades quietly when the interpreter is not configured...
    assert build_diarizer("auto") is None
    # ...but an explicit request must fail loudly rather than silently no-op.
    with pytest.raises(ValueError, match="SLIDEGEIST_DIARIZEN_PYTHON"):
        build_diarizer("local")
    with pytest.raises(ValueError, match="Unknown diarization mode"):
        build_diarizer("banana")


def test_env_var_sets_the_default_mode(monkeypatch) -> None:
    """A batch driver that cannot pass flags must still be able to opt out."""
    monkeypatch.delenv("SLIDEGEIST_DIARIZE", raising=False)
    assert diarize.get_diarize_mode() == "auto"

    monkeypatch.setenv("SLIDEGEIST_DIARIZE", "off")
    assert diarize.get_diarize_mode() == "off"

    monkeypatch.setenv("SLIDEGEIST_DIARIZE", "  OFF  ")
    assert diarize.get_diarize_mode() == "off"


def test_unknown_env_mode_fails_before_any_work(monkeypatch) -> None:
    """A typo must not silently leave diarization on for a whole batch."""
    monkeypatch.setenv("SLIDEGEIST_DIARIZE", "none")

    with pytest.raises(ValueError, match="SLIDEGEIST_DIARIZE"):
        diarize.get_diarize_mode()


def test_gpu_oom_raises_instead_of_falling_back_to_cpu(monkeypatch, tmp_path) -> None:
    """An OOM on a 32 GB machine is a misconfiguration, not a reason to go slow."""
    calls: list[str] = []

    def fake_run(command, **kwargs):
        calls.append(command[command.index("--device") + 1])
        return subprocess.CompletedProcess(command, 1, "", "CUDA error: out of memory")

    monkeypatch.setattr(subprocess, "run", fake_run)
    diarizer = DiariZenDiarizer(interpreter="/usr/bin/python3.11", model="m", device="cuda")

    with pytest.raises(RuntimeError, match="Not falling back to CPU"):
        diarizer.diarize(tmp_path / "audio.wav")

    assert calls == ["cuda"], "must not silently retry on CPU"
