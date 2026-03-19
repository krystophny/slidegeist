"""Slidegeist: Extract slides and transcripts from lecture videos."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("slidegeist")
except PackageNotFoundError:  # pragma: no cover - occurs only in editable installs
    __version__ = "0+unknown"
