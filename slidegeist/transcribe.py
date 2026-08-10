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


class _WordRequired(TypedDict):
    """Required word fields."""

    word: str
    start: float
    end: float


class Word(_WordRequired, total=False):
    """A single word with timing and an optional speaker label."""

    speaker: str


class _SegmentRequired(TypedDict):
    """Required segment fields."""

    start: float
    end: float
    text: str
    words: list[Word]


class SpeakerShare(TypedDict):
    """How much of a segment a speaker occupies."""

    speaker: str
    seconds: float


class Segment(_SegmentRequired, total=False):
    """A transcript segment with timing, words and optional speaker labels.

    ``speakers`` is written only when more than one speaker overlaps the
    segment, so single-speaker material stays byte-identical to schema v1.
    """

    speaker: str
    speakers: list[SpeakerShare]


class TranscriptResult(TypedDict):
    """Complete transcription result."""

    language: str
    segments: list[Segment]


def _normalize_transcript(payload: dict[str, object]) -> TranscriptResult:
    """Normalize a whisper.cpp response into Slidegeist's format."""
    return _normalize_segments(payload, drop_incompatible_word_timing=True, speaker_key=None)


def _normalize_voxtral_transcript(payload: dict[str, object]) -> TranscriptResult:
    """Normalize a Voxtral response, keeping its word clock and speaker labels.

    Voxtral reports word and segment times on the same clock, so the whisper.cpp
    VAD workaround must not run here: applying it would silently downgrade
    word-level diarization to segment level.
    """
    return _normalize_segments(
        payload, drop_incompatible_word_timing=False, speaker_key="speaker_id"
    )


_WORD_CLOCK_TOLERANCE_S = 0.25


def _rebase_word_timing(segments: list[Segment]) -> int:
    """Put word times back on the segment clock, returning how many were rebased.

    whisper.cpp with VAD enabled reports segment times on the original audio
    timeline but word times on the silence-compressed one. The two clocks differ
    by however much silence VAD removed before the segment.

    Segment bounds are trustworthy, so each segment's words are mapped affinely
    from their own span onto the segment's span. That is exact whenever VAD
    removed no silence *inside* the segment (the common case, since VAD cuts at
    speech boundaries) and stays bounded by the segment length otherwise. Words
    that cannot be trusted at all - non-finite, inverted or non-monotonic - are
    dropped for that segment alone rather than for the whole response.
    """
    rebased = 0
    for segment in segments:
        words = segment["words"]
        if not words:
            continue

        seg_start = segment["start"]
        seg_end = segment["end"]

        usable = True
        previous_end = -math.inf
        for word in words:
            if (
                not math.isfinite(word["start"])
                or not math.isfinite(word["end"])
                or word["end"] < word["start"]
                or word["start"] < previous_end - _WORD_CLOCK_TOLERANCE_S
            ):
                usable = False
                break
            previous_end = word["end"]

        if not usable:
            logger.warning(
                "Dropping unusable word timing for segment at %.2fs-%.2fs", seg_start, seg_end
            )
            segment["words"] = []
            continue

        span_start = min(word["start"] for word in words)
        span_end = max(word["end"] for word in words)

        within_segment = (
            span_start >= seg_start - _WORD_CLOCK_TOLERANCE_S
            and span_end <= seg_end + _WORD_CLOCK_TOLERANCE_S
        )
        if within_segment:
            continue

        span = span_end - span_start
        seg_span = seg_end - seg_start
        if span <= 0 or seg_span <= 0:
            # Degenerate: nothing to interpolate against.
            segment["words"] = []
            continue

        scale = seg_span / span
        for word in words:
            word["start"] = min(seg_end, max(seg_start, seg_start + (word["start"] - span_start) * scale))
            word["end"] = min(seg_end, max(seg_start, seg_start + (word["end"] - span_start) * scale))
        rebased += 1

    return rebased


def _normalize_segments(
    payload: dict[str, object],
    *,
    drop_incompatible_word_timing: bool,
    speaker_key: str | None,
) -> TranscriptResult:
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
                    entry: Word = {
                        "word": word_text,
                        "start": word_start,
                        "end": word_end,
                    }
                    if speaker_key is not None:
                        word_speaker = word.get(speaker_key)
                        if word_speaker not in (None, ""):
                            entry["speaker"] = str(word_speaker)
                    words.append(entry)

            record: Segment = {
                "start": start,
                "end": end,
                "text": text,
                "words": words,
            }
            if speaker_key is not None:
                segment_speaker = segment.get(speaker_key)
                if segment_speaker not in (None, ""):
                    record["speaker"] = str(segment_speaker)
            segments.append(record)

    if drop_incompatible_word_timing:
        remapped = _rebase_word_timing(segments)
        if remapped:
            logger.warning(
                "Rebased word timing for %d/%d segments: whisper.cpp with VAD reports "
                "word times on the silence-compressed clock while segment times follow "
                "the original audio",
                remapped,
                len(segments),
            )

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
