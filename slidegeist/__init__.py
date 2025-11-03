"""Slidegeist: Extract slides and transcripts from lecture videos."""

import os
from importlib.metadata import PackageNotFoundError, version

# Set NumExpr threads to actual CPU count for optimal performance
# This prevents the "NUMEXPR_MAX_THREADS is set to safe value" warning
# and allows full utilization of available cores
if "NUMEXPR_MAX_THREADS" not in os.environ:
    os.environ["NUMEXPR_MAX_THREADS"] = str(os.cpu_count() or 16)

try:
    __version__ = version("slidegeist")
except PackageNotFoundError:  # pragma: no cover - occurs only in editable installs
    __version__ = "0+unknown"
