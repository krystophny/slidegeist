"""Video download from URLs using yt-dlp."""

import logging
import re
import tempfile
from pathlib import Path
from typing import Any, Literal

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
