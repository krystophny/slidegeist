"""Transcription backends.

Voxtral (Mistral, remote) is the default; the local Whisper server stays
available through ``--local``. There is deliberately no automatic fallback
between them: silently switching would either upload audio the user meant to
keep local, or bill for a run they expected to be free.
"""

from __future__ import annotations

import logging
from pathlib import Path
from tempfile import TemporaryDirectory

from slidegeist.constants import (
    DEFAULT_TRANSCRIBER,
    DEFAULT_VOXTRAL_MAX_CHUNK_S,
    DEFAULT_WHISPER_MODEL,
)
from slidegeist.diarize import BaseDiarizer, SpeakerTurn, assign_speakers, canonicalize_speaker_ids
from slidegeist.ffmpeg import extract_audio_compressed, get_video_duration
from slidegeist.services import get_voxtral_model, voxtral_transcribe
from slidegeist.transcribe import (
    CHUNK_DURATION_S,
    Segment,
    TranscriptResult,
    _chunk_start_offsets,
    _normalize_voxtral_transcript,
    _split_audio_chunks,
    transcribe_video,
)

logger = logging.getLogger(__name__)


class BaseTranscriber:
    """Interface for transcription backends."""

    name: str = "unknown"
    provider: str = "unknown"
    model: str = ""
    provides_speakers: bool = False

    def transcribe(
        self,
        video_path: Path,
        *,
        work_dir: Path | None = None,
        diarizer: BaseDiarizer | None = None,
    ) -> tuple[TranscriptResult, list[SpeakerTurn]]:
        raise NotImplementedError


class WhisperCppTranscriber(BaseTranscriber):
    """Local Whisper-compatible HTTP server (whisper.cpp and friends)."""

    provider = "whisper.cpp"

    def __init__(self, model: str | None = None) -> None:
        self.model = model or DEFAULT_WHISPER_MODEL
        self.name = f"{self.model} (whisper.cpp)"
        self.chunk_duration = CHUNK_DURATION_S

    def transcribe(
        self,
        video_path: Path,
        *,
        work_dir: Path | None = None,
        diarizer: BaseDiarizer | None = None,
    ) -> tuple[TranscriptResult, list[SpeakerTurn]]:
        turns: list[SpeakerTurn] = []
        cache = (work_dir / "diarization.json") if work_dir is not None else None
        result = transcribe_video(
            video_path,
            model_size=self.model,
            diarizer=diarizer,
            diarization_cache=cache,
            speaker_turns=turns,
        )
        if turns:
            assign_speakers(result["segments"], turns)
            canonicalize_speaker_ids(result["segments"], turns)
        return result, turns


class VoxtralTranscriber(BaseTranscriber):
    """Mistral Voxtral, which returns speaker labels and word timing natively."""

    provider = "voxtral"
    provides_speakers = True

    def __init__(self, model: str | None = None, max_chunk_s: int | None = None) -> None:
        self.model = model or get_voxtral_model()
        self.name = f"{self.model} (Mistral)"
        self.max_chunk_s = int(max_chunk_s or DEFAULT_VOXTRAL_MAX_CHUNK_S)

    def transcribe(
        self,
        video_path: Path,
        *,
        work_dir: Path | None = None,
        diarizer: BaseDiarizer | None = None,
    ) -> tuple[TranscriptResult, list[SpeakerTurn]]:
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        try:
            duration = get_video_duration(video_path)
        except Exception as exc:  # pragma: no cover - probe failures are logged only
            duration = None
            logger.warning("Could not determine duration before transcription: %s", exc)

        if duration:
            logger.info(
                "Sending %.1f minutes of audio to %s (billed per audio minute)",
                duration / 60.0,
                self.name,
            )

        turns: list[SpeakerTurn] = []
        with TemporaryDirectory(prefix="slidegeist-voxtral-") as temp_dir:
            temp = Path(temp_dir)

            if diarizer is not None:
                from slidegeist.ffmpeg import extract_audio

                wav_path = temp / f"{video_path.stem}.wav"
                extract_audio(video_path, wav_path)
                cache = (work_dir / "diarization.json") if work_dir is not None else None
                turns.extend(diarizer.diarize(wav_path, cache_path=cache))

            audio_path = temp / f"{video_path.stem}.ogg"
            extract_audio_compressed(video_path, audio_path)

            if duration is None or duration <= self.max_chunk_s:
                # The common lecture case: one request, no segmentation at all.
                payloads = [(audio_path, 0.0)]
            else:
                chunks = _split_audio_chunks(
                    audio_path, temp / "chunks", chunk_duration=self.max_chunk_s, suffix="ogg"
                )
                payloads = list(zip(chunks, _chunk_start_offsets(chunks), strict=True))
                logger.warning(
                    "Audio exceeds %ds and was split into %d requests; Voxtral clusters "
                    "speakers per request, so cross-request speaker identity is unresolved. "
                    "Use --diarize local for globally consistent labels.",
                    self.max_chunk_s,
                    len(payloads),
                )

            all_segments: list[Segment] = []
            language = "unknown"
            for index, (chunk_path, offset) in enumerate(payloads):
                payload = voxtral_transcribe(chunk_path, model=self.model, diarize=True)
                chunk = _normalize_voxtral_transcript(payload)
                if chunk["language"] != "unknown":
                    language = chunk["language"]

                namespace = f"c{index:02d}/" if len(payloads) > 1 else ""
                for segment in chunk["segments"]:
                    segment["start"] += offset
                    segment["end"] += offset
                    if namespace and segment.get("speaker"):
                        segment["speaker"] = f"{namespace}{segment['speaker']}"
                    for word in segment.get("words", []):
                        word["start"] += offset
                        word["end"] += offset
                        if namespace and word.get("speaker"):
                            word["speaker"] = f"{namespace}{word['speaker']}"
                    all_segments.append(segment)

        result: TranscriptResult = {"language": language, "segments": all_segments}

        if turns:
            # An explicit local diarization overrides the provider's own labels,
            # which is the documented fix for chunk-local speaker identity.
            assign_speakers(result["segments"], turns)

        canonicalize_speaker_ids(result["segments"], turns)
        logger.info("Voxtral transcription complete: %d segments", len(all_segments))
        return result, turns


def build_transcriber(provider: str = DEFAULT_TRANSCRIBER, model: str | None = None) -> BaseTranscriber:
    """Build a transcription backend by provider name."""
    if provider in ("whisper", "whisper.cpp", "local"):
        return WhisperCppTranscriber(model)
    if provider in ("voxtral", "mistral"):
        return VoxtralTranscriber(model)
    raise ValueError(f"Unknown transcription provider: {provider!r}")


__all__ = [
    "BaseTranscriber",
    "VoxtralTranscriber",
    "WhisperCppTranscriber",
    "build_transcriber",
]
