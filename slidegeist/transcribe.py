"""Audio transcription through the local voxtype OpenAI-compatible service."""

from __future__ import annotations

import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypedDict

from slidegeist.constants import DEFAULT_WHISPER_MODEL
from slidegeist.ffmpeg import extract_audio, get_video_duration
from slidegeist.services import voxtype_transcribe

logger = logging.getLogger(__name__)


class Word(TypedDict):
    """A single word with timing information."""

    word: str
    start: float
    end: float


class Segment(TypedDict):
    """A transcript segment with timing and words."""

    start: float
    end: float
    text: str
    words: list[Word]


class TranscriptResult(TypedDict):
    """Complete transcription result."""

    language: str
    segments: list[Segment]


def _normalize_transcript(payload: dict[str, object]) -> TranscriptResult:
    """Normalize an OpenAI-style STT response into Slidegeist's format."""
    segments_payload = payload.get("segments", [])
    segments: list[Segment] = []

    if isinstance(segments_payload, list):
        for segment in segments_payload:
            if not isinstance(segment, dict):
                continue

            text = str(segment.get("text", "")).strip()
            if not text:
                continue

            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", start))
            words: list[Word] = []
            words_payload = segment.get("words", [])

            if isinstance(words_payload, list):
                for word in words_payload:
                    if not isinstance(word, dict):
                        continue
                    word_text = str(word.get("word", "")).strip()
                    if not word_text:
                        continue
                    words.append(
                        {
                            "word": word_text,
                            "start": float(word.get("start", start)),
                            "end": float(word.get("end", end)),
                        }
                    )

            segments.append(
                {
                    "start": start,
                    "end": end,
                    "text": text,
                    "words": words,
                }
            )

    if not segments:
        text = str(payload.get("text", "")).strip()
        if text:
            duration = float(payload.get("duration", 0.0))
            segments.append(
                {
                    "start": 0.0,
                    "end": duration,
                    "text": text,
                    "words": [],
                }
            )

    return {
        "language": str(payload.get("language", "unknown")),
        "segments": segments,
    }


def transcribe_video(
    video_path: Path,
    model_size: str = DEFAULT_WHISPER_MODEL,
) -> TranscriptResult:
    """Extract audio and transcribe it with the configured voxtype service."""

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        video_duration = get_video_duration(video_path)
    except Exception as exc:
        video_duration = None
        logger.warning("Could not determine video duration before transcription: %s", exc)
    else:
        logger.info(
            "Video duration: %.1f minutes (%.1f seconds)",
            video_duration / 60.0,
            video_duration,
        )

    with TemporaryDirectory(prefix="slidegeist-voxtype-") as temp_dir:
        audio_path = Path(temp_dir) / f"{video_path.stem}.wav"
        extract_audio(video_path, audio_path)
        logger.info("Submitting audio to voxtype with model %s", model_size)
        payload = voxtype_transcribe(audio_path, model=model_size)

    result = _normalize_transcript(payload)
    logger.info("Voxtype transcription complete: %d segments", len(result["segments"]))
    return result
