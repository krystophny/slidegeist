"""Speaker diarization and the merge of speaker turns onto transcript timing.

Diarization answers "how many voices, and when did each speak". Labels are
session-local: ``SPEAKER_00`` in one recording is not the same person as
``SPEAKER_00`` in the next. Recognising a person across recordings is speaker
*identification* and needs an enrollment step that is deliberately not built
here.

The local backend (DiariZen) runs in a separate interpreter: torch has no wheels
for the 3.14 host interpreter, and DiariZen's pretrained weights are CC BY-NC
4.0, so they must not become a bundled dependency of this MIT-licensed package.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import TypedDict

from slidegeist.constants import (
    DEFAULT_DIARIZE_MODE,
    DEFAULT_DIARIZEN_MODEL,
    DEFAULT_DIARIZEN_TIMEOUT,
)
from slidegeist.transcribe import Segment, SpeakerShare

logger = logging.getLogger(__name__)

DIARIZATION_SCHEMA = "slidegeist-diarization-v1"
_NEAREST_TURN_TOLERANCE_S = 0.5


class SpeakerTurn(TypedDict):
    """A stretch of audio attributed to one speaker."""

    start: float
    end: float
    speaker: str


class BaseDiarizer:
    """Interface for speaker diarization backends."""

    name: str = "unknown"
    provider: str = "unknown"

    def diarize(self, audio_path: Path, *, cache_path: Path | None = None) -> list[SpeakerTurn]:
        raise NotImplementedError


class DiariZenDiarizer(BaseDiarizer):
    """Run DiariZen in a separate interpreter and collect its speaker turns."""

    provider = "diarizen"

    def __init__(
        self,
        interpreter: str | None = None,
        model: str | None = None,
        device: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.interpreter = interpreter or os.getenv("SLIDEGEIST_DIARIZEN_PYTHON", "").strip()
        if not self.interpreter:
            raise ValueError(
                "SLIDEGEIST_DIARIZEN_PYTHON is not set; point it at a Python "
                "interpreter that has DiariZen installed"
            )
        self.model = model or os.getenv("SLIDEGEIST_DIARIZEN_MODEL", DEFAULT_DIARIZEN_MODEL)
        self.device = device or os.getenv("SLIDEGEIST_DIARIZEN_DEVICE", "auto")
        raw_timeout = os.getenv("SLIDEGEIST_DIARIZEN_TIMEOUT", str(DEFAULT_DIARIZEN_TIMEOUT))
        try:
            self.timeout = float(timeout if timeout is not None else raw_timeout)
        except ValueError as exc:
            raise ValueError("SLIDEGEIST_DIARIZEN_TIMEOUT must be a number") from exc
        self.name = f"{self.model} (DiariZen)"

    @property
    def _worker_path(self) -> Path:
        return Path(__file__).with_name("_diarizen_worker.py")

    def diarize(self, audio_path: Path, *, cache_path: Path | None = None) -> list[SpeakerTurn]:
        if cache_path is not None and cache_path.exists():
            cached = self._load_cache(cache_path)
            if cached is not None:
                logger.info("Reusing cached diarization from %s", cache_path)
                return cached

        turns = self._run_worker(audio_path, device=self.device)

        if cache_path is not None:
            payload = {
                "schema": DIARIZATION_SCHEMA,
                "model": self.model,
                "audio_duration": max((turn["end"] for turn in turns), default=0.0),
                "turns": turns,
            }
            cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        return turns

    def _load_cache(self, cache_path: Path) -> list[SpeakerTurn] | None:
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Ignoring unreadable diarization cache %s", cache_path)
            return None
        if payload.get("schema") != DIARIZATION_SCHEMA or payload.get("model") != self.model:
            logger.info("Diarization cache does not match the current model; recomputing")
            return None
        turns = payload.get("turns")
        return turns if isinstance(turns, list) else None

    def _run_worker(self, audio_path: Path, *, device: str) -> list[SpeakerTurn]:
        command = [
            self.interpreter,
            "-u",
            str(self._worker_path),
            "--audio",
            str(audio_path),
            "--model",
            self.model,
            "--device",
            device,
        ]
        logger.info("Diarizing %s with %s", audio_path.name, self.name)
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=self.timeout, check=False
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"DiariZen timed out after {self.timeout:.0f}s") from exc

        stderr_tail = "\n".join((completed.stderr or "").strip().splitlines()[-40:])

        if completed.returncode != 0:
            if device != "cpu" and "out of memory" in (completed.stderr or "").lower():
                logger.warning("DiariZen ran out of GPU memory; retrying on CPU")
                return self._run_worker(audio_path, device="cpu")
            raise RuntimeError(f"DiariZen failed (exit {completed.returncode}):\n{stderr_tail}")

        if stderr_tail:
            logger.debug("DiariZen stderr:\n%s", stderr_tail)

        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"DiariZen produced unparsable output:\n{stderr_tail}") from exc

        if payload.get("schema") != DIARIZATION_SCHEMA:
            raise RuntimeError(f"Unexpected diarization schema: {payload.get('schema')!r}")

        turns = payload.get("turns", [])
        if not isinstance(turns, list):
            raise RuntimeError("Diarization payload has no turn list")
        logger.info("Diarization found %d turns", len(turns))
        return turns


def build_diarizer(mode: str = DEFAULT_DIARIZE_MODE) -> BaseDiarizer | None:
    """Build a diarizer for the requested mode.

    ``off`` disables diarization, ``provider`` defers to the transcriber's own
    speaker labels, ``local`` requires DiariZen and fails loudly when it is not
    configured, and ``auto`` uses DiariZen when available.
    """
    if mode in ("off", "provider"):
        return None
    if mode == "local":
        return DiariZenDiarizer()
    if mode == "auto":
        interpreter = os.getenv("SLIDEGEIST_DIARIZEN_PYTHON", "").strip()
        if not interpreter or not Path(interpreter).exists():
            logger.info(
                "Local diarization unavailable (SLIDEGEIST_DIARIZEN_PYTHON unset or missing); "
                "continuing without it"
            )
            return None
        return DiariZenDiarizer(interpreter=interpreter)
    raise ValueError(f"Unknown diarization mode: {mode!r}")


def _overlap_by_speaker(
    start: float, end: float, turns: list[SpeakerTurn], hint: int = 0
) -> tuple[dict[str, float], int]:
    """Accumulate overlap seconds per speaker over [start, end).

    ``hint`` is the index of the first turn that may overlap; it is returned
    updated so a caller sweeping forward stays linear rather than quadratic.
    """
    totals: dict[str, float] = {}
    index = hint
    while index < len(turns) and turns[index]["end"] <= start:
        index += 1

    cursor = index
    while cursor < len(turns) and turns[cursor]["start"] < end:
        turn = turns[cursor]
        overlap = min(end, turn["end"]) - max(start, turn["start"])
        if overlap > 0:
            totals[turn["speaker"]] = totals.get(turn["speaker"], 0.0) + overlap
        cursor += 1

    return totals, index


def _nearest_speaker(start: float, end: float, turns: list[SpeakerTurn]) -> str | None:
    """Return the speaker of the closest turn within the tolerance window."""
    best: str | None = None
    best_gap = _NEAREST_TURN_TOLERANCE_S
    for turn in turns:
        if turn["end"] < start:
            gap = start - turn["end"]
        elif turn["start"] > end:
            gap = turn["start"] - end
        else:
            return turn["speaker"]
        if gap <= best_gap:
            best_gap = gap
            best = turn["speaker"]
    return best


def assign_speakers(segments: list[Segment], turns: list[SpeakerTurn]) -> None:
    """Attach speaker labels to segments and words, in place.

    Words are assigned by dominant overlap. A segment takes the speaker holding
    the most *word time* rather than the most words, so one long technical term
    cannot be outvoted by a run of short function words. Segments without word
    timing - the whisper VAD path, or a provider that returns segments only -
    are assigned by segment overlap, which is a first-class path rather than a
    fallback.
    """
    if not turns:
        return

    ordered = sorted(turns, key=lambda turn: (turn["start"], turn["end"]))
    hint = 0

    for segment in segments:
        words = segment.get("words") or []
        per_speaker: dict[str, float] = {}

        if words:
            for word in words:
                totals, _ = _overlap_by_speaker(word["start"], word["end"], ordered, hint)
                if totals:
                    speaker = max(totals.items(), key=lambda item: item[1])[0]
                    word["speaker"] = speaker
                    per_speaker[speaker] = per_speaker.get(speaker, 0.0) + (
                        word["end"] - word["start"]
                    )
                else:
                    nearest = _nearest_speaker(word["start"], word["end"], ordered)
                    if nearest is not None:
                        word["speaker"] = nearest
                        per_speaker[nearest] = per_speaker.get(nearest, 0.0) + (
                            word["end"] - word["start"]
                        )

        segment_totals, hint = _overlap_by_speaker(
            segment["start"], segment["end"], ordered, hint
        )

        if not words:
            per_speaker = segment_totals

        if per_speaker:
            segment["speaker"] = max(per_speaker.items(), key=lambda item: item[1])[0]

        if len(segment_totals) > 1:
            shares: list[SpeakerShare] = [
                {"speaker": speaker, "seconds": round(seconds, 3)}
                for speaker, seconds in sorted(
                    segment_totals.items(), key=lambda item: item[1], reverse=True
                )
            ]
            segment["speakers"] = shares


def canonicalize_speaker_ids(segments: list[Segment], turns: list[SpeakerTurn]) -> list[str]:
    """Renumber speakers to SPEAKER_00... in order of first appearance.

    Providers label speakers differently (``speaker_0``, ``SPEAKER_02``, a
    chunk-namespaced id). Canonical ids keep transcripts comparable across
    providers, which a later ensemble mode needs in order to align tracks.
    """
    mapping: dict[str, str] = {}

    def canonical(label: str) -> str:
        if label not in mapping:
            mapping[label] = f"SPEAKER_{len(mapping):02d}"
        return mapping[label]

    for segment in segments:
        if segment.get("speaker"):
            segment["speaker"] = canonical(segment["speaker"])
        for word in segment.get("words") or []:
            if word.get("speaker"):
                word["speaker"] = canonical(word["speaker"])
        shares = segment.get("speakers")
        if shares:
            for share in shares:
                share["speaker"] = canonical(share["speaker"])

    for turn in turns:
        turn["speaker"] = canonical(turn["speaker"])

    return list(mapping.values())


__all__ = [
    "BaseDiarizer",
    "DiariZenDiarizer",
    "SpeakerTurn",
    "assign_speakers",
    "build_diarizer",
    "canonicalize_speaker_ids",
]
