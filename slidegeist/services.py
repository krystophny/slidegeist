"""Helpers for talking to locally hosted AI services."""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import uuid
from functools import lru_cache
from http.client import HTTPConnection, HTTPSConnection, HTTPResponse
from pathlib import Path
from typing import Any
from urllib import error, parse, request

from slidegeist.constants import DEFAULT_LLAMACPP_URL, DEFAULT_WHISPER_URL

logger = logging.getLogger(__name__)


def get_llama_cpp_url() -> str:
    """Return the configured llama.cpp base URL."""
    return os.getenv("SLIDEGEIST_LLAMACPP_URL", DEFAULT_LLAMACPP_URL).rstrip("/")


def get_whisper_url() -> str:
    """Return the configured Whisper server base URL."""
    return os.getenv("SLIDEGEIST_WHISPER_URL", DEFAULT_WHISPER_URL).rstrip("/")


def _http_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 5.0,
) -> dict[str, Any]:
    """Send a JSON HTTP request and decode the JSON response."""
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    http_request = request.Request(url, data=data, headers=headers, method=method)
    with request.urlopen(http_request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw) if raw else {}


def _http_status(url: str, timeout: float = 2.0) -> int:
    """Return the HTTP status code for a lightweight probe request."""
    try:
        with request.urlopen(url, timeout=timeout) as response:
            return response.status
    except error.HTTPError as exc:
        return exc.code
    except error.URLError:
        return 0


def is_llama_cpp_available(timeout: float = 2.0) -> bool:
    """Check whether the configured llama.cpp server is reachable."""
    return _http_status(f"{get_llama_cpp_url()}/health", timeout=timeout) == 200


def is_whisper_available(timeout: float = 2.0) -> bool:
    """Check whether the configured Whisper transcription server is reachable."""
    status = _http_status(f"{get_whisper_url()}/v1/audio/transcriptions", timeout=timeout)
    return status in {200, 405}


@lru_cache(maxsize=4)
def get_llama_cpp_model(base_url: str | None = None) -> str | None:
    """Return the first advertised llama.cpp model id."""
    resolved_base_url = (base_url or get_llama_cpp_url()).rstrip("/")
    try:
        payload = _http_json(f"{resolved_base_url}/v1/models", timeout=5.0)
    except (OSError, ValueError, error.URLError) as exc:
        logger.debug("Could not query llama.cpp models: %s", exc)
        return None

    data = payload.get("data", [])
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                model_id = item.get("id")
                if isinstance(model_id, str) and model_id:
                    return model_id

    models = payload.get("models", [])
    if isinstance(models, list):
        for item in models:
            if isinstance(item, dict):
                model_id = item.get("model") or item.get("name")
                if isinstance(model_id, str) and model_id:
                    return model_id

    return None


def llama_cpp_complete(
    prompt: str,
    *,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    timeout: float = 180.0,
) -> str:
    """Run a text completion against llama.cpp's OpenAI-compatible API."""
    payload: dict[str, Any] = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    model = get_llama_cpp_model()
    if model:
        payload["model"] = model

    response = _http_json(
        f"{get_llama_cpp_url()}/v1/completions",
        method="POST",
        payload=payload,
        timeout=timeout,
    )
    choices = response.get("choices", [])
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("llama.cpp returned no completion choices")

    choice = choices[0]
    if not isinstance(choice, dict):
        raise RuntimeError("llama.cpp returned an invalid completion payload")

    text = choice.get("text", "")
    if not isinstance(text, str):
        raise RuntimeError("llama.cpp completion text was not a string")

    return text.strip()


def _build_multipart_prefix(
    boundary: str,
    fields: list[tuple[str, str]],
    file_field: str,
    file_path: Path,
) -> tuple[bytes, bytes]:
    """Create multipart prefix and suffix for a streaming upload."""
    lines: list[bytes] = []

    for key, value in fields:
        lines.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )

    mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    lines.extend(
        [
            f"--{boundary}\r\n".encode("utf-8"),
            (
                f'Content-Disposition: form-data; name="{file_field}"; '
                f'filename="{file_path.name}"\r\n'
            ).encode("utf-8"),
            f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8"),
        ]
    )

    prefix = b"".join(lines)
    suffix = f"\r\n--{boundary}--\r\n".encode("utf-8")
    return prefix, suffix


def _connection_for(url: parse.SplitResult, timeout: float) -> HTTPConnection | HTTPSConnection:
    """Create an HTTP or HTTPS connection for the given parsed URL."""
    port = url.port
    if url.scheme == "https":
        return HTTPSConnection(url.hostname, port, timeout=timeout)
    return HTTPConnection(url.hostname, port, timeout=timeout)


def _read_response(response: HTTPResponse) -> dict[str, Any]:
    """Read and decode a JSON response body."""
    raw = response.read().decode("utf-8")
    return json.loads(raw) if raw else {}


def whisper_transcribe(
    audio_path: Path,
    *,
    model: str,
    language: str = "auto",
    timeout: float = 1800.0,
) -> dict[str, Any]:
    """Transcribe an audio file through the Whisper-compatible HTTP API."""
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    endpoint = parse.urlsplit(f"{get_whisper_url()}/v1/audio/transcriptions")
    if not endpoint.hostname:
        raise RuntimeError("Whisper URL is missing a hostname")

    boundary = f"slidegeist-{uuid.uuid4().hex}"
    fields = [
        ("model", model),
        ("language", language),
        ("response_format", "verbose_json"),
        ("timestamp_granularities[]", "segment"),
        ("timestamp_granularities[]", "word"),
    ]
    prefix, suffix = _build_multipart_prefix(boundary, fields, "file", audio_path)
    content_length = len(prefix) + audio_path.stat().st_size + len(suffix)

    path = endpoint.path or "/"
    if endpoint.query:
        path = f"{path}?{endpoint.query}"

    connection = _connection_for(endpoint, timeout)
    try:
        connection.putrequest("POST", path)
        connection.putheader("Content-Type", f"multipart/form-data; boundary={boundary}")
        connection.putheader("Content-Length", str(content_length))
        connection.endheaders()

        connection.send(prefix)
        with audio_path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                connection.send(chunk)
        connection.send(suffix)

        response = connection.getresponse()
        payload = _read_response(response)
        if response.status >= 400:
            raise RuntimeError(f"whisper HTTP {response.status}: {payload}")
        return payload
    finally:
        connection.close()
