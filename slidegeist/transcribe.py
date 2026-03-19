"""Audio transcription via Voxtype STT service (OpenAI-compatible API)."""

import json
import logging
import time
from pathlib import Path
from typing import TypedDict
from urllib.request import Request, urlopen

from slidegeist.constants import DEFAULT_DEVICE, DEFAULT_STT_URL, DEFAULT_WHISPER_MODEL

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


def _build_multipart(
    fields: dict[str, str],
    file_field: str,
    file_path: Path,
    file_content_type: str = "audio/wav",
) -> tuple[bytes, str]:
    """Build multipart/form-data body for urllib.

    Args:
        fields: Simple key-value form fields.
        file_field: Name of the file field.
        file_path: Path to the file to upload.
        file_content_type: MIME type for the file.

    Returns:
        Tuple of (body_bytes, content_type_header).
    """
    boundary = "----SlidegeistBoundary9876543210"
    parts: list[bytes] = []

    for key, value in fields.items():
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode())
        parts.append(f"{value}\r\n".encode())

    parts.append(f"--{boundary}\r\n".encode())
    parts.append(
        f'Content-Disposition: form-data; name="{file_field}"; '
        f'filename="{file_path.name}"\r\n'.encode()
    )
    parts.append(f"Content-Type: {file_content_type}\r\n\r\n".encode())
    parts.append(file_path.read_bytes())
    parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())

    body = b"".join(parts)
    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


def _is_service_available(stt_url: str) -> bool:
    """Check if the STT service is reachable."""
    try:
        req = Request(f"{stt_url}/v1/models", method="GET")
        with urlopen(req, timeout=2):
            return True
    except Exception:
        # Try health endpoint as fallback
        try:
            req = Request(f"{stt_url}/health", method="GET")
            with urlopen(req, timeout=2):
                return True
        except Exception:
            return False


def _transcribe_via_service(
    audio_path: Path,
    stt_url: str,
    model: str,
    language: str | None = None,
) -> TranscriptResult:
    """Transcribe audio via Voxtype OpenAI-compatible STT API.

    Args:
        audio_path: Path to audio file (WAV format).
        stt_url: Base URL of the STT service.
        model: Whisper model name.
        language: Language code (e.g., 'en', 'de') or None for auto-detection.

    Returns:
        TranscriptResult with segments and detected language.
    """
    fields: dict[str, str] = {
        "model": model,
        "response_format": "verbose_json",
        "timestamp_granularities[]": "word",
    }
    if language:
        fields["language"] = language

    body, content_type = _build_multipart(fields, "file", audio_path)

    url = f"{stt_url}/v1/audio/transcriptions"
    req = Request(url, data=body, method="POST")
    req.add_header("Content-Type", content_type)

    logger.info(f"Sending audio to STT service at {url}")
    start_time = time.time()

    with urlopen(req, timeout=7200) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    elapsed = time.time() - start_time
    logger.info(f"STT service responded in {elapsed:.1f}s")

    segments_list: list[Segment] = []
    detected_language = result.get("language", "unknown")

    for seg in result.get("segments", []):
        words_list: list[Word] = []
        for word in seg.get("words", []):
            words_list.append({
                "word": word.get("word", ""),
                "start": float(word.get("start", 0.0)),
                "end": float(word.get("end", 0.0)),
            })
        segments_list.append({
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", 0.0)),
            "text": seg.get("text", "").strip(),
            "words": words_list,
        })

    # If no segments from API, try top-level words
    if not segments_list and result.get("words"):
        all_words: list[Word] = []
        for word in result["words"]:
            all_words.append({
                "word": word.get("word", ""),
                "start": float(word.get("start", 0.0)),
                "end": float(word.get("end", 0.0)),
            })
        if all_words:
            full_text = result.get("text", " ".join(w["word"] for w in all_words)).strip()
            segments_list.append({
                "start": all_words[0]["start"],
                "end": all_words[-1]["end"],
                "text": full_text,
                "words": all_words,
            })

    logger.info(f"Transcription complete: {len(segments_list)} segments, language={detected_language}")
    return {"language": detected_language, "segments": segments_list}


def transcribe_video(
    video_path: Path,
    model_size: str = DEFAULT_WHISPER_MODEL,
    device: str = DEFAULT_DEVICE,
    compute_type: str = "int8",
    stt_url: str = DEFAULT_STT_URL,
) -> TranscriptResult:
    """Transcribe video audio using the Voxtype STT service.

    Extracts audio from the video, then sends it to the STT service.
    The service must be running (see scripts/setup-voxtype-stt.sh).

    Args:
        video_path: Path to the video file.
        model_size: Whisper model name (default: large-v3-turbo).
        device: Unused, kept for API compatibility.
        compute_type: Unused, kept for API compatibility.
        stt_url: Base URL of the Voxtype STT service.

    Returns:
        Dictionary with language and segments containing timestamped text.

    Raises:
        FileNotFoundError: If video file does not exist.
        ConnectionError: If STT service is not reachable.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Check service availability
    if not _is_service_available(stt_url):
        raise ConnectionError(
            f"STT service not available at {stt_url}\n"
            f"Start the service with: scripts/setup-voxtype-stt.sh\n"
            f"Or install voxtype and run:\n"
            f"  voxtype --service --service-host 127.0.0.1 --service-port 8427 "
            f"--model {model_size} daemon"
        )

    # Extract audio to a temporary WAV file
    import tempfile

    from slidegeist.ffmpeg import extract_audio

    with tempfile.TemporaryDirectory() as tmp_dir:
        audio_path = Path(tmp_dir) / "audio.wav"
        logger.info(f"Extracting audio from {video_path.name}")
        extract_audio(video_path, audio_path)

        return _transcribe_via_service(audio_path, stt_url, model_size)
