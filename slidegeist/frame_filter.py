"""Filter non-instructional visual states after multimodal classification."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any

_NON_SLIDE = re.compile(
    r"0\.\s*FRAME TYPE\s*:?\s*(?:\n|\s)*NON[-_ ]?SLIDE\b",
    re.IGNORECASE,
)


def is_non_slide_description(description: str) -> bool:
    """Return whether a structured description classifies the frame as non-slide."""
    return bool(_NON_SLIDE.search(description))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def filter_non_slide_states(
    slide_metadata: list[tuple[int, float, float, Path]],
    descriptions: dict[str, str],
    diagnostics_path: Path,
) -> list[tuple[int, float, float, Path]]:
    """Reject classified desktop/navigation states and merge their time spans.

    Rejected images are reversibly renamed with a ``.non-slide`` suffix. The
    raw detector boundaries, classification, and image hash remain in the
    transition diagnostic so the filtering decision is auditable.
    """
    rejected = [
        item
        for item in slide_metadata
        if is_non_slide_description(descriptions.get(item[3].stem, ""))
    ]
    if not rejected:
        return slide_metadata

    accepted = [item for item in slide_metadata if item not in rejected]
    if not accepted:
        raise RuntimeError("multimodal classification rejected every extracted frame")

    payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("transition diagnostics must be a JSON object")
    current_timestamps = payload.get("raw_timestamps", payload.get("timestamps"))
    if not isinstance(current_timestamps, list):
        raise RuntimeError("transition diagnostics have no timestamp list")
    if len(current_timestamps) != len(slide_metadata) - 1:
        raise RuntimeError(
            "cannot filter frames: transition and extracted-state counts disagree"
        )

    video_end = slide_metadata[-1][2]
    filtered: list[tuple[int, float, float, Path]] = []
    for position, (index, start, _end, image_path) in enumerate(accepted):
        filtered_start = 0.0 if position == 0 else start
        filtered_end = (
            accepted[position + 1][1]
            if position + 1 < len(accepted)
            else video_end
        )
        filtered.append((index, filtered_start, filtered_end, image_path))

    payload["raw_timestamps"] = current_timestamps
    payload["timestamps"] = [item[1] for item in filtered[1:]]
    payload["rejected_states"] = [
        {
            "slide_id": image_path.stem,
            "start": start,
            "end": end,
            "image_sha256": _sha256(image_path),
            "classification": descriptions[image_path.stem],
        }
        for _index, start, end, image_path in rejected
    ]
    payload["state_filter"] = "multimodal-instructional-content-v1"
    _write_json(diagnostics_path, payload)

    for _index, _start, _end, image_path in rejected:
        image_path.rename(image_path.with_suffix(f"{image_path.suffix}.non-slide"))

    return filtered
