"""Audio transcription through a local OpenAI-compatible Whisper server."""

from __future__ import annotations

import logging
import math
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypedDict

from slidegeist.constants import DEFAULT_WHISPER_MODEL
from slidegeist.ffmpeg import extract_audio, get_video_duration
from slidegeist.services import whisper_transcribe

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
            if not math.isfinite(start) or not math.isfinite(end) or end < start:
                logger.warning("Discarding transcript segment with invalid timing")
                continue
            words: list[Word] = []
            words_payload = segment.get("words", [])

            if isinstance(words_payload, list):
                for word in words_payload:
                    if not isinstance(word, dict):
                        continue
                    word_text = str(word.get("word", "")).strip()
                    if not word_text:
                        continue
                    word_start = float(word.get("start", start))
                    word_end = float(word.get("end", end))
                    words.append(
                        {
                            "word": word_text,
                            "start": word_start,
                            "end": word_end,
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

    # whisper.cpp with VAD currently reports word times on the concatenated
    # speech timeline while segment times refer to the original audio. A mixed
    # time base is worse than no word timing: consumers may align words to the
    # wrong slide. If any word violates its segment, discard word timing for the
    # complete response while preserving the independently timed segments.
    previous_word_end = -math.inf
    invalid_word_timing = False
    for segment in segments:
        for word in segment["words"]:
            word_start = word["start"]
            word_end = word["end"]
            if (
                not math.isfinite(word_start)
                or not math.isfinite(word_end)
                or word_end < word_start
                or word_start < segment["start"] - 0.25
                or word_end > segment["end"] + 0.25
                or word_start < previous_word_end - 0.25
            ):
                invalid_word_timing = True
                break
            previous_word_end = word_end
        if invalid_word_timing:
            break

    if invalid_word_timing:
        logger.warning(
            "Whisper returned word and segment timestamps on incompatible time bases; "
            "discarding unreliable word timing"
        )
        for segment in segments:
            segment["words"] = []

    if not segments:
        text = str(payload.get("text", "")).strip()
        if text:
            raw_duration = payload.get("duration", 0.0)
            duration = float(raw_duration) if isinstance(raw_duration, (int, float, str)) else 0.0
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


CHUNK_DURATION_S = 120  # 2-minute chunks to stay within server upload limits


def _split_audio_chunks(
    audio_path: Path, chunk_dir: Path, chunk_duration: int = CHUNK_DURATION_S
) -> list[Path]:
    """Split a WAV file into fixed-length chunks using ffmpeg segment muxer."""
    import subprocess as _sp

    chunk_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(chunk_dir / "chunk_%04d.wav")
    cmd = [
        "ffmpeg",
        "-i",
        str(audio_path),
        "-f",
        "segment",
        "-segment_time",
        str(chunk_duration),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-y",
        pattern,
    ]
    _sp.run(cmd, check=True, capture_output=True, text=True)
    chunks = sorted(chunk_dir.glob("chunk_*.wav"))
    logger.info("Split audio into %d chunks of %ds each", len(chunks), chunk_duration)
    return chunks


def _chunk_start_offsets(chunks: list[Path]) -> list[float]:
    """Measure cumulative PCM sample time instead of assuming exact segment cuts."""
    offsets = []
    offset = 0.0
    for chunk in chunks:
        offsets.append(offset)
        try:
            with wave.open(str(chunk), "rb") as stream:
                frame_rate = stream.getframerate()
                duration = stream.getnframes() / frame_rate
        except (OSError, EOFError, wave.Error, ZeroDivisionError):
            logger.warning(
                "Could not measure %s; falling back to nominal chunk duration",
                chunk,
            )
            duration = float(CHUNK_DURATION_S)
        offset += duration
    return offsets


def transcribe_video(
    video_path: Path,
    model_size: str = DEFAULT_WHISPER_MODEL,
) -> TranscriptResult:
    """Extract audio and transcribe it via the configured Whisper HTTP server.

    Long audio is automatically split into 2-minute chunks to stay within
    server upload size limits, then reassembled with corrected timestamps.
    """

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        video_duration = get_video_duration(video_path)
    except Exception as exc:
        video_duration = None
        logger.warning("Could not determine video duration before transcription: %s", exc)
    else:
        if video_duration is not None:
            logger.info(
                "Video duration: %.1f minutes (%.1f seconds)",
                video_duration / 60.0,
                video_duration,
            )

    with TemporaryDirectory(prefix="slidegeist-whisper-") as temp_dir:
        temp = Path(temp_dir)
        audio_path = temp / f"{video_path.stem}.wav"
        extract_audio(video_path, audio_path)

        chunks = _split_audio_chunks(audio_path, temp / "chunks")

        all_segments: list[Segment] = []
        detected_language = "unknown"

        for idx, (chunk_path, offset) in enumerate(
            zip(chunks, _chunk_start_offsets(chunks), strict=True)
        ):
            logger.info(
                "Transcribing chunk %d/%d (offset %.0fs) with model %s",
                idx + 1,
                len(chunks),
                offset,
                model_size,
            )
            try:
                payload = whisper_transcribe(chunk_path, model=model_size)
            except Exception as exc:
                raise RuntimeError(
                    f"Whisper chunk {idx + 1}/{len(chunks)} failed; "
                    "refusing to publish a partial transcript"
                ) from exc

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
    logger.info(
        "Whisper transcription complete: %d segments from %d chunks", len(all_segments), len(chunks)
    )
    return result
