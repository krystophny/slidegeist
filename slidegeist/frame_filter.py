"""Filter non-instructional visual states after multimodal classification."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any

_FRAME_TYPE = re.compile(
    r"^0\.\s*FRAME TYPE\s*:?\s*(?:\n[ \t]*)?(NON[-_ ]?SLIDE|SLIDE)\b",
    re.IGNORECASE | re.MULTILINE,
)
_STATE_FILTER = "multimodal-instructional-content-v1"


def frame_type(description: str) -> str | None:
    """Return the unambiguous structured frame type, if one is present."""
    matches = {
        match.group(1).upper().replace("_", "-").replace(" ", "-")
        for match in _FRAME_TYPE.finditer(description)
    }
    return matches.pop() if len(matches) == 1 else None


def is_slide_description(description: str) -> bool:
    """Return whether a structured description explicitly accepts the frame."""
    return frame_type(description) == "SLIDE"


def is_non_slide_description(description: str) -> bool:
    """Return whether a structured description explicitly rejects the frame."""
    return frame_type(description) == "NON-SLIDE"


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


def _filtered_metadata(
    slide_metadata: list[tuple[int, float, float, Path]],
    rejected_ids: set[str],
    video_end: float,
) -> list[tuple[int, float, float, Path]]:
    accepted = [item for item in slide_metadata if item[3].stem not in rejected_ids]
    if not accepted:
        raise RuntimeError("multimodal classification rejected every extracted frame")

    filtered: list[tuple[int, float, float, Path]] = []
    for position, (index, start, _end, image_path) in enumerate(accepted):
        filtered_start = 0.0 if position == 0 else start
        filtered_end = (
            accepted[position + 1][1]
            if position + 1 < len(accepted)
            else video_end
        )
        filtered.append((index, filtered_start, filtered_end, image_path))
    return filtered


def _complete_rejected_renames(
    slide_metadata: list[tuple[int, float, float, Path]],
    rejected_states: list[dict[str, Any]],
) -> int:
    """Finish the reversible renames described by a durable filter plan."""
    if not slide_metadata:
        raise RuntimeError("cannot recover state filtering without extracted frames")
    slides_dir = slide_metadata[0][3].parent
    by_id = {item[3].stem: item[3] for item in slide_metadata}
    already_renamed = 0

    for state in rejected_states:
        slide_id = state.get("slide_id")
        expected_hash = state.get("image_sha256")
        if not isinstance(slide_id, str) or not isinstance(expected_hash, str):
            raise RuntimeError("state-filter diagnostic has an invalid rejection record")

        source = by_id.get(slide_id)
        targets = list(slides_dir.glob(f"{slide_id}.*.non-slide"))
        if len(targets) > 1:
            raise RuntimeError(f"multiple rejected-frame files found for {slide_id}")
        target = targets[0] if targets else None
        if source is not None and target is not None:
            raise RuntimeError(f"both live and rejected-frame files exist for {slide_id}")
        if source is None and target is None:
            raise RuntimeError(f"rejected-frame file is missing for {slide_id}")

        candidate = source or target
        assert candidate is not None
        if _sha256(candidate) != expected_hash:
            raise RuntimeError(f"rejected-frame checksum mismatch for {slide_id}")
        if source is not None:
            source.rename(source.with_suffix(f"{source.suffix}.non-slide"))
        else:
            already_renamed += 1

    return already_renamed


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
    payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("transition diagnostics must be a JSON object")

    existing_rejections = payload.get("rejected_states")
    if payload.get("state_filter") == _STATE_FILTER:
        if not isinstance(existing_rejections, list):
            raise RuntimeError("state-filter diagnostic has no rejection plan")
        raw_timestamps = payload.get("raw_timestamps")
        video_end = payload.get("state_filter_video_end")
        if not isinstance(raw_timestamps, list) or not isinstance(
            video_end, (int, float)
        ):
            raise RuntimeError("state-filter diagnostic is incomplete")
        already_renamed = _complete_rejected_renames(
            slide_metadata, existing_rejections
        )
        if len(slide_metadata) + already_renamed != len(raw_timestamps) + 1:
            raise RuntimeError(
                "cannot recover filtering: detector and frame counts disagree"
            )
        rejected_ids = {str(state["slide_id"]) for state in existing_rejections}
        return _filtered_metadata(slide_metadata, rejected_ids, float(video_end))

    unclassified = [
        item[3].stem
        for item in slide_metadata
        if frame_type(descriptions.get(item[3].stem, "")) is None
    ]
    if unclassified:
        raise RuntimeError(
            "multimodal descriptions lack an unambiguous frame type: "
            + ", ".join(unclassified)
        )

    rejected = [
        item
        for item in slide_metadata
        if is_non_slide_description(descriptions.get(item[3].stem, ""))
    ]
    current_timestamps = payload.get("timestamps")
    if not isinstance(current_timestamps, list):
        raise RuntimeError("transition diagnostics have no timestamp list")
    if len(current_timestamps) != len(slide_metadata) - 1:
        raise RuntimeError(
            "cannot filter frames: transition and extracted-state counts disagree"
        )

    video_end = slide_metadata[-1][2]
    rejected_ids = {item[3].stem for item in rejected}
    filtered = _filtered_metadata(slide_metadata, rejected_ids, video_end)

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
    payload["state_filter"] = _STATE_FILTER
    payload["state_filter_video_end"] = video_end
    _write_json(diagnostics_path, payload)

    _complete_rejected_renames(slide_metadata, payload["rejected_states"])

    return filtered
