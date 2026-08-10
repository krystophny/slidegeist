"""Video download from URLs using yt-dlp, with a direct Opencast path."""

import json
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Literal
from urllib import request

import yt_dlp  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

BrowserType = Literal[
    "firefox", "safari", "chrome", "chromium", "edge", "opera", "brave"
]


def translate_url(url: str) -> str:
    """Translate URLs to formats compatible with yt-dlp extractors.

    Args:
        url: Original URL.

    Returns:
        Translated URL if translation needed, otherwise original URL.

    Examples:
        TU Graz portal URL -> paella URL:
        https://tube.tugraz.at/portal/watch/<UUID>
        -> https://tube.tugraz.at/paella/ui/watch.html?id=<UUID>
    """
    # TU Graz Tube: portal format -> paella format
    tugraz_portal_pattern = r"https?://tube\.tugraz\.at/portal/watch/([0-9a-fA-F-]+)"
    match = re.match(tugraz_portal_pattern, url)
    if match:
        video_id = match.group(1)
        translated = f"https://tube.tugraz.at/paella/ui/watch.html?id={video_id}"
        logger.info(f"Translated TU Graz URL: {url} -> {translated}")
        return translated

    return url


OPENCAST_EVENT_RE = re.compile(
    r"https?://(?P<host>[\w.-]+)/(?:play|portal/watch|paella/ui/watch\.html\?id=)/?(?P<id>[0-9a-fA-F-]{36})"
)


def _opencast_cookie_header(host: str) -> str | None:
    """Return a Cookie header from a cached Opencast session, if one exists.

    Recorded lectures on an institutional Opencast are usually behind SSO. A
    session cached by another tool (helpy's ``tube_session.json``) is reused
    rather than re-implementing the login.
    """
    path = Path(
        os.getenv("SLIDEGEIST_OPENCAST_SESSION", Path.home() / ".config/helpy/tube_session.json")
    ).expanduser()
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        jar = payload.get("cookies", {}).get(host) or []
        pairs = [f"{c['name']}={c['value']}" for c in jar if c.get("name")]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None
    return "; ".join(pairs) or None


def resolve_opencast_track(url: str) -> tuple[str, str | None] | None:
    """Resolve an Opencast event URL to a directly downloadable media URL.

    Opencast exposes the same recording through two APIs that disagree. The
    external API (``/api/events/<id>/publications``) reports ``media: []`` for
    a number of older events, while the search API
    (``/search/episode.json?id=<id>``) still lists concrete MP4 tracks for
    exactly those events. The search API is therefore tried as a fallback.

    Returns (media_url, cookie_header) or None when nothing is downloadable.
    A progressive MP4 is preferred over HLS: it downloads at whatever speed the
    server allows instead of being paced by the stream.
    """
    match = OPENCAST_EVENT_RE.search(url)
    if not match:
        return None
    host, event_id = match.group("host"), match.group("id")
    cookie = _opencast_cookie_header(host)
    headers = {"Cookie": cookie} if cookie else {}

    def _get_json(endpoint: str) -> Any:
        request_obj = request.Request(f"https://{host}/{endpoint}", headers=headers)
        try:
            with request.urlopen(request_obj, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception:  # noqa: BLE001 - any failure just means "try the next source"
            return None

    def _tracks(payload: Any) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []

        def walk(node: Any) -> None:
            if isinstance(node, dict):
                if isinstance(node.get("url"), str):
                    found.append(node)
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)

        walk(payload)
        return found

    candidates: list[dict[str, Any]] = []
    for endpoint in (
        f"api/events/{event_id}/publications",
        f"search/episode.json?id={event_id}",
    ):
        candidates.extend(_tracks(_get_json(endpoint)))
        progressive = [
            t for t in candidates
            if str(t.get("url", "")).endswith(".mp4") and "/static/" in str(t.get("url", ""))
        ]
        if progressive:
            def _size(track: dict[str, Any]) -> int:
                try:
                    return int(track.get("size") or 0)
                except (TypeError, ValueError):
                    return 0

            best = max(progressive, key=_size)
            logger.info("Resolved Opencast event %s to a progressive MP4", event_id[:8])
            return str(best["url"]), cookie

    streaming = [t for t in candidates if ".m3u8" in str(t.get("url", ""))]
    if streaming:
        logger.info(
            "Opencast event %s has no progressive MP4; falling back to HLS", event_id[:8]
        )
        return str(streaming[0]["url"]), cookie
    return None


def download_opencast(url: str, output_dir: Path) -> Path | None:
    """Download an Opencast recording as fast as the server allows.

    ffmpeg copies the streams without re-encoding and without ``-re``, so an
    HLS source is fetched segment-by-segment at network speed rather than being
    played back in real time.
    """
    resolved = resolve_opencast_track(url)
    if resolved is None:
        return None
    media_url, cookie = resolved
    output_dir.mkdir(parents=True, exist_ok=True)
    event_id = OPENCAST_EVENT_RE.search(url).group("id")  # type: ignore[union-attr]
    destination = output_dir / f"{event_id}.mp4"
    if destination.exists() and destination.stat().st_size > 0:
        logger.info("Reusing existing download %s", destination.name)
        return destination

    command = ["ffmpeg", "-nostdin", "-loglevel", "error"]
    if cookie:
        command += ["-headers", f"Cookie: {cookie}\r\n"]
    command += ["-i", media_url, "-c", "copy", "-movflags", "+faststart", "-y", str(destination)]
    logger.info("Downloading Opencast media with ffmpeg (stream copy, no realtime pacing)")
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not destination.exists() or destination.stat().st_size == 0:
        destination.unlink(missing_ok=True)
        raise ValueError(
            f"Opencast download failed for {event_id}: "
            f"{(completed.stderr or '').strip()[-400:]}"
        )
    return destination


def check_existing_video(url: str, output_dir: Path, cookies_from_browser: BrowserType | None = None) -> Path | None:
    """Check if video from URL already exists in output directory.

    Args:
        url: Video URL.
        output_dir: Directory to check for existing video.
        cookies_from_browser: Browser to extract cookies from (needed for metadata extraction).

    Returns:
        Path to existing video file if found, None otherwise.
    """
    if not output_dir.exists():
        return None

    # Translate URL first
    url = translate_url(url)

    # Extract video metadata without downloading
    ydl_opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
    }

    if cookies_from_browser:
        ydl_opts["cookiesfrombrowser"] = (cookies_from_browser,)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if info is None:
                return None

            # Construct expected filename
            sanitized_title = ydl.prepare_filename(info)
            expected_filename = Path(sanitized_title).name

            # Check if file exists in output directory
            potential_path = output_dir / expected_filename
            if potential_path.exists():
                logger.info(f"Found existing video: {potential_path}")
                return potential_path

    except Exception as e:
        logger.debug(f"Could not check for existing video: {e}")
        return None

    return None


def download_video(
    url: str,
    output_dir: Path | None = None,
    cookies_from_browser: BrowserType | None = None
) -> Path:
    """Download video from URL using yt-dlp, or reuse if already downloaded.

    Supports YouTube, Mediasite, TU Graz Tube, and many other platforms.
    If output_dir is specified and video already exists there, reuses it.

    Args:
        url: Video URL to download.
        output_dir: Directory to save video. If None, creates a temporary directory
            with prefix 'slidegeist_'. Caller is responsible for cleanup of temp files.
        cookies_from_browser: Browser to extract cookies from for authentication.
            Supports: firefox, safari, chrome, chromium, edge, opera, brave.

    Returns:
        Path to the downloaded (or existing) video file.

    Raises:
        ValueError: If video information cannot be extracted from URL.
        FileNotFoundError: If downloaded file cannot be found after download.
        RuntimeError: If download fails for other reasons (network, permissions, etc.).

    Examples:
        # Public video
        video_path = download_video("https://youtube.com/watch?v=...")

        # Authenticated video using Firefox cookies
        video_path = download_video(
            "https://tube.tugraz.at/...",
            cookies_from_browser="firefox"
        )
    """
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="slidegeist_"))
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check if video already exists
        existing = check_existing_video(url, output_dir, cookies_from_browser)
        if existing:
            logger.info(f"Reusing existing video: {existing}")
            return existing

    # Template for output filename: use video title, sanitized
    output_template = str(output_dir / "%(title)s.%(ext)s")

    # yt-dlp options
    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": output_template,
        "quiet": False,
        "no_warnings": False,
        "extract_flat": False,
        "merge_output_format": "mp4",
    }

    # Opencast first: yt-dlp cannot see tracks that the external API hides, and
    # the search API exposes a progressive MP4 for exactly those events.
    try:
        direct = download_opencast(url, output_dir)
    except ValueError as exc:
        logger.warning("Opencast direct download failed (%s); falling back to yt-dlp", exc)
        direct = None
    if direct is not None:
        return direct

    # Translate URL to yt-dlp-compatible format if needed
    url = translate_url(url)

    # Add browser cookies if specified
    if cookies_from_browser:
        logger.info(f"Using cookies from {cookies_from_browser} browser")
        ydl_opts["cookiesfrombrowser"] = (cookies_from_browser,)

    logger.info(f"Downloading video from: {url}")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract info first to get the final filename
            info = ydl.extract_info(url, download=True)

            if info is None:
                raise ValueError(f"Failed to extract video information from URL: {url}")

            # Get the downloaded file path
            if "requested_downloads" in info and info["requested_downloads"]:
                downloaded_file = Path(info["requested_downloads"][0]["filepath"])
            else:
                # Fallback: construct filename from info
                sanitized_title = ydl.prepare_filename(info)
                downloaded_file = Path(sanitized_title)

            if not downloaded_file.exists():
                raise FileNotFoundError(f"Downloaded file not found: {downloaded_file}")

            logger.info(f"Downloaded video to: {downloaded_file}")
            return downloaded_file

    except FileNotFoundError:
        # Re-raise FileNotFoundError as-is
        raise
    except ValueError:
        # Re-raise ValueError as-is
        raise
    except Exception as e:
        # Wrap other exceptions with context
        logger.error(f"Failed to download video from {url}: {e}")
        raise RuntimeError(f"Video download failed: {e}") from e


def get_video_filename(url: str, cookies_from_browser: BrowserType | None = None) -> str:
    """Get the filename that would be used for a video URL without downloading.

    Args:
        url: Video URL.
        cookies_from_browser: Browser to extract cookies from (needed for authenticated videos).

    Returns:
        Filename (stem without extension) that will be used for the video.

    Raises:
        ValueError: If video information cannot be extracted.
    """
    url = translate_url(url)

    ydl_opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
    }

    if cookies_from_browser:
        ydl_opts["cookiesfrombrowser"] = (cookies_from_browser,)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if info is None:
                raise ValueError(f"Failed to extract video information from URL: {url}")

            # Get sanitized filename
            sanitized_title = ydl.prepare_filename(info)
            return Path(sanitized_title).stem

    except Exception as e:
        raise ValueError(f"Failed to get video filename from {url}: {e}") from e


def is_url(input_str: str) -> bool:
    """Check if input string is a URL.

    Args:
        input_str: String to check.

    Returns:
        True if input looks like a URL, False otherwise.
    """
    return (
        input_str.startswith(("http://", "https://", "www."))
        and len(input_str) > 7  # Minimum valid URL: "http://x" or "www.x.c"
    )
