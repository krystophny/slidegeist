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


CHUNK_DURATION_S = 120  # 2-minute chunks to stay within voxtype upload limits


def _split_audio_chunks(audio_path: Path, chunk_dir: Path,
                        chunk_duration: int = CHUNK_DURATION_S) -> list[Path]:
    """Split a WAV file into fixed-length chunks using ffmpeg segment muxer."""
    import subprocess as _sp

    chunk_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(chunk_dir / "chunk_%04d.wav")
    cmd = [
        "ffmpeg", "-i", str(audio_path),
        "-f", "segment", "-segment_time", str(chunk_duration),
        "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le",
        "-y", pattern,
    ]
    _sp.run(cmd, check=True, capture_output=True, text=True)
    chunks = sorted(chunk_dir.glob("chunk_*.wav"))
    logger.info("Split audio into %d chunks of %ds each", len(chunks), chunk_duration)
    return chunks


def transcribe_video(
    video_path: Path,
    model_size: str = DEFAULT_WHISPER_MODEL,
) -> TranscriptResult:
    """Extract audio and transcribe it with the configured voxtype service.

    Long audio is automatically split into 2-minute chunks to stay within
    voxtype upload size limits, then reassembled with corrected timestamps.
    """

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
        temp = Path(temp_dir)
        audio_path = temp / f"{video_path.stem}.wav"
        extract_audio(video_path, audio_path)

        chunks = _split_audio_chunks(audio_path, temp / "chunks")

        all_segments: list[Segment] = []
        detected_language = "unknown"

        for idx, chunk_path in enumerate(chunks):
            offset = idx * CHUNK_DURATION_S
            logger.info(
                "Transcribing chunk %d/%d (offset %.0fs) with model %s",
                idx + 1, len(chunks), offset, model_size,
            )
            try:
                payload = voxtype_transcribe(chunk_path, model=model_size)
            except Exception as exc:
                logger.warning("Chunk %d failed: %s — skipping", idx + 1, exc)
                continue

            chunk_result = _normalize_transcript(payload)
            if chunk_result["language"] != "unknown":
                detected_language = chunk_result["language"]

            for seg in chunk_result["segments"]:
                seg["start"] += offset
                seg["end"] += offset
                for w in seg.get("words", []):
                    w["start"] += offset
                    w["end"] += offset
                all_segments.append(seg)

    result: TranscriptResult = {
        "language": detected_language,
        "segments": all_segments,
    }
    logger.info("Voxtype transcription complete: %d segments from %d chunks",
                len(all_segments), len(chunks))
    return result
