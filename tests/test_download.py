"""Tests for video download functionality."""

from slidegeist.download import is_url, translate_url


def test_is_url_http() -> None:
    """Test URL detection for http URLs."""
    assert is_url("http://example.com/video.mp4")


def test_is_url_https() -> None:
    """Test URL detection for https URLs."""
    assert is_url("https://youtube.com/watch?v=123")


def test_is_url_www() -> None:
    """Test URL detection for www URLs."""
    assert is_url("www.example.com")


def test_is_url_file_path() -> None:
    """Test URL detection returns False for file paths."""
    assert not is_url("/path/to/video.mp4")
    assert not is_url("video.mp4")
    assert not is_url("../relative/path.mp4")


def test_translate_url_tugraz_portal_to_paella() -> None:
    """Test TU Graz portal URL is translated to paella format."""
    portal_url = "https://tube.tugraz.at/portal/watch/ab28ec60-8cbe-4f1a-9b96-a95add56c612"
    expected = "https://tube.tugraz.at/paella/ui/watch.html?id=ab28ec60-8cbe-4f1a-9b96-a95add56c612"
    assert translate_url(portal_url) == expected


def test_translate_url_tugraz_portal_http() -> None:
    """Test TU Graz portal URL works with http (not just https)."""
    portal_url = "http://tube.tugraz.at/portal/watch/ab28ec60-8cbe-4f1a-9b96-a95add56c612"
    result = translate_url(portal_url)
    assert "paella/ui/watch.html?id=ab28ec60-8cbe-4f1a-9b96-a95add56c612" in result


def test_translate_url_tugraz_paella_unchanged() -> None:
    """Test TU Graz paella URL passes through unchanged."""
    paella_url = "https://tube.tugraz.at/paella/ui/watch.html?id=ab28ec60-8cbe-4f1a-9b96-a95add56c612"
    assert translate_url(paella_url) == paella_url


def test_translate_url_youtube_unchanged() -> None:
    """Test YouTube URLs pass through unchanged."""
    youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert translate_url(youtube_url) == youtube_url


def test_translate_url_mediasite_unchanged() -> None:
    """Test Mediasite URLs pass through unchanged."""
    mediasite_url = "https://iaea.mediasite.com/Mediasite/Play/2943a8bc00c74c53a055136c2e4a1f851d"
    assert translate_url(mediasite_url) == mediasite_url


def test_translate_url_other_urls_unchanged() -> None:
    """Test other URLs pass through unchanged."""
    urls = [
        "https://example.com/video.mp4",
        "http://vimeo.com/123456789",
        "https://dailymotion.com/video/xyz",
    ]
    for url in urls:
        assert translate_url(url) == url
