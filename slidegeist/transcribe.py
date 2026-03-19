"""Audio transcription via Voxtype STT service (OpenAI-compatible API)."""

import json
import logging
import socket
import time
from pathlib import Path
from typing import TypedDict
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from slidegeist.constants import DEFAULT_DEVICE, DEFAULT_STT_URL, DEFAULT_WHISPER_MODEL

logger = logging.getLogger(__name__)

# Chunk duration in seconds for splitting long audio files.
# Voxtype has an HTTP body size limit, so we split audio into manageable pieces.
CHUNK_DURATION_SECS = 300  # 5 minutes


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
    """Check if the STT service is reachable by attempting a TCP connection."""
    parsed = urlparse(stt_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 80
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except (OSError, TimeoutError):
        return False


def _transcribe_chunk(
    audio_path: Path,
    stt_url: str,
    model: str,
) -> dict[str, object]:
    """Transcribe a single audio chunk via the STT API.

    Args:
        audio_path: Path to audio file (WAV format).
        stt_url: Base URL of the STT service.
        model: Whisper model name.

    Returns:
        Parsed JSON response dict with text, segments, duration, language.
    """
    fields: dict[str, str] = {
        "model": model,
        "response_format": "verbose_json",
    }

    body, content_type = _build_multipart(fields, "file", audio_path)

    url = f"{stt_url}/v1/audio/transcriptions"
    req = Request(url, data=body, method="POST")
    req.add_header("Content-Type", content_type)

    with urlopen(req, timeout=3600) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _split_audio_into_chunks(
    audio_path: Path,
    chunk_dir: Path,
    chunk_duration: int = CHUNK_DURATION_SECS,
) -> list[tuple[Path, float]]:
    """Split audio file into chunks using FFmpeg.

    Args:
        audio_path: Path to full audio file.
        chunk_dir: Directory to write chunks into.
        chunk_duration: Duration of each chunk in seconds.

    Returns:
        List of (chunk_path, start_offset_seconds) tuples.
    """
    import subprocess

    # Get total duration
    from slidegeist.ffmpeg import get_video_duration

    total_duration = get_video_duration(audio_path)
    chunks: list[tuple[Path, float]] = []

    offset = 0.0
    chunk_idx = 0
    while offset < total_duration:
        chunk_path = chunk_dir / f"chunk_{chunk_idx:04d}.wav"
        cmd = [
            "ffmpeg", "-y", "-i", str(audio_path),
            "-ss", str(offset),
            "-t", str(chunk_duration),
            "-ar", "16000", "-ac", "1",
            "-f", "wav", str(chunk_path),
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        chunks.append((chunk_path, offset))
        offset += chunk_duration
        chunk_idx += 1

    return chunks


def transcribe_video(
    video_path: Path,
    model_size: str = DEFAULT_WHISPER_MODEL,
    device: str = DEFAULT_DEVICE,
    compute_type: str = "int8",
    stt_url: str = DEFAULT_STT_URL,
) -> TranscriptResult:
    """Transcribe video audio using the Voxtype STT service.

    Extracts audio from the video, splits into chunks if needed, then sends
    each chunk to the STT service. Results are combined with correct timestamps.

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

    if not _is_service_available(stt_url):
        raise ConnectionError(
            f"STT service not available at {stt_url}\n"
            f"Start the service with: scripts/setup-voxtype-stt.sh\n"
            f"Or install voxtype and run:\n"
            f"  voxtype --service --service-host 127.0.0.1 --service-port 8427 "
            f"--model {model_size} daemon"
        )

    import tempfile

    from tqdm import tqdm

    from slidegeist.ffmpeg import extract_audio, get_video_duration

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        audio_path = tmp_path / "audio.wav"
        logger.info(f"Extracting audio from {video_path.name}")
        extract_audio(video_path, audio_path)

        total_duration = get_video_duration(video_path)
        audio_size_mb = audio_path.stat().st_size / (1024 * 1024)
        logger.info(
            f"Audio: {total_duration / 60:.1f} min, {audio_size_mb:.0f} MB"
        )

        # Split into chunks if audio is longer than chunk duration
        if total_duration > CHUNK_DURATION_SECS:
            logger.info(
                f"Splitting audio into {CHUNK_DURATION_SECS}s chunks "
                f"({int(total_duration / CHUNK_DURATION_SECS) + 1} chunks)"
            )
            chunks = _split_audio_into_chunks(audio_path, tmp_path)
        else:
            chunks = [(audio_path, 0.0)]

        segments_list: list[Segment] = []
        start_time = time.time()

        detected_language = "auto"
        for chunk_path, chunk_offset in tqdm(chunks, desc="Transcribing", unit="chunk"):
            logger.info(
                f"Sending chunk at {chunk_offset:.0f}s to STT service"
            )
            result = _transcribe_chunk(chunk_path, stt_url, model_size)

            if result.get("language"):
                detected_language = result["language"]

            if "segments" in result and result["segments"]:
                for seg in result["segments"]:
                    words_list: list[Word] = []
                    for word in seg.get("words", []):
                        words_list.append({
                            "word": word.get("word", ""),
                            "start": float(word.get("start", 0.0)) + chunk_offset,
                            "end": float(word.get("end", 0.0)) + chunk_offset,
                        })
                    segments_list.append({
                        "start": float(seg.get("start", 0.0)) + chunk_offset,
                        "end": float(seg.get("end", 0.0)) + chunk_offset,
                        "text": seg.get("text", "").strip(),
                        "words": words_list,
                    })
            elif result.get("text", "").strip():
                chunk_duration = result.get("duration", CHUNK_DURATION_SECS)
                segments_list.append({
                    "start": chunk_offset,
                    "end": chunk_offset + float(chunk_duration),
                    "text": result["text"].strip(),
                    "words": [],
                })

        elapsed = time.time() - start_time
        speed_factor = total_duration / elapsed if elapsed > 0 else 0
        logger.info(
            f"Transcription complete: {len(segments_list)} segments in "
            f"{elapsed / 60:.1f}min ({speed_factor:.1f}x realtime)"
        )

        return {"language": detected_language, "segments": segments_list}
