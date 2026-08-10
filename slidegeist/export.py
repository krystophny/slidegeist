"""Export slide metadata to Markdown files."""

from __future__ import annotations

import logging
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slidegeist.ai_description import BaseSlideDescriber
from slidegeist.ocr import OcrPipeline
from slidegeist.transcribe import Segment

logger = logging.getLogger(__name__)


def _parse_existing_markdown(markdown_path: Path) -> dict[str, dict[str, str]]:
    """Parse existing slides.md to extract per-slide content.

    Args:
        markdown_path: Path to existing slides.md file.

    Returns:
        Dictionary mapping slide_id to dict with keys: 'transcript', 'ocr', 'visual_elements'.
        Returns empty dict if file doesn't exist or can't be parsed.
    """
    if not markdown_path.exists():
        return {}

    if markdown_path.name == "index.md":
        split_data: dict[str, dict[str, str]] = {}
        for slide_path in sorted(markdown_path.parent.glob("slide_*.md")):
            split_data.update(_parse_split_slide_markdown(slide_path))
        return split_data

    try:
        content = markdown_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning(f"Could not read existing markdown: {e}")
        return {}

    slides_data: dict[str, dict[str, str]] = {}
    current_slide_id: str | None = None
    current_section: str | None = None
    current_content: list[str] = []

    for line in content.split("\n"):
        # Detect slide anchor: <a name="slide_001"></a>
        if line.strip().startswith('<a name="slide_'):
            if current_slide_id and current_section:
                # Save previous section
                slides_data.setdefault(current_slide_id, {})
                slides_data[current_slide_id][current_section] = "\n".join(current_content).strip()
                current_content = []

            # Extract slide ID
            try:
                current_slide_id = line.split('name="')[1].split('"')[0]
                slides_data.setdefault(
                    current_slide_id,
                    {"transcript": "", "ocr": "", "visual_elements": "", "ai_description": ""},
                )
                current_section = None
            except Exception:
                current_slide_id = None
            continue

        # Detect sections
        if line.strip() == "### Transcript":
            if current_slide_id and current_section:
                slides_data[current_slide_id][current_section] = "\n".join(current_content).strip()
            current_section = "transcript"
            current_content = []
            continue

        if line.strip() == "### OCR Text":
            if current_slide_id and current_section:
                slides_data[current_slide_id][current_section] = "\n".join(current_content).strip()
            current_section = "ocr"
            current_content = []
            continue

        if line.strip().startswith("**Visual Elements:**"):
            if current_slide_id and current_section:
                slides_data[current_slide_id][current_section] = "\n".join(current_content).strip()
            current_section = "visual_elements"
            current_content = [line]  # Include the header
            continue

        if line.strip() == "### AI Description (for reconstruction)":
            if current_slide_id and current_section:
                slides_data[current_slide_id][current_section] = "\n".join(current_content).strip()
            current_section = "ai_description"
            current_content = []
            continue

        # Skip separator lines
        if line.strip() == "---":
            if current_slide_id and current_section:
                slides_data[current_slide_id][current_section] = "\n".join(current_content).strip()
            current_section = None
            current_content = []
            continue

        # Accumulate content
        if current_slide_id and current_section:
            current_content.append(line)

    # Save last section
    if current_slide_id and current_section:
        slides_data[current_slide_id][current_section] = "\n".join(current_content).strip()

    return slides_data


def _parse_split_slide_markdown(path: Path) -> dict[str, dict[str, str]]:
    """Parse one YAML-frontmatter slide file from split export mode."""
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not read existing split slide %s: %s", path, exc)
        return {}
    identifier = re.search(r"^id:\s*(\S+)\s*$", content, re.MULTILINE)
    if identifier is None:
        return {}
    data = {
        "transcript": "",
        "ocr": "",
        "visual_elements": "",
        "ai_description": "",
    }
    headings = list(
        re.finditer(
            r"^## (Transcript|OCR Text|AI Description \(for reconstruction\))\s*$",
            content,
            re.MULTILINE,
        )
    )
    keys = {
        "Transcript": "transcript",
        "OCR Text": "ocr",
        "AI Description (for reconstruction)": "ai_description",
    }
    for position, heading in enumerate(headings):
        end = headings[position + 1].start() if position + 1 < len(headings) else len(content)
        data[keys[heading.group(1)]] = content[heading.end() : end].strip()
    visual = re.search(r"^\*\*Visual Elements:\*\*(.*)$", content, re.MULTILINE)
    if visual:
        data["visual_elements"] = visual.group(0).strip()
    return {identifier.group(1): data}


def _atomic_write_text(path: Path, content: str) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def export_slides_json(
    video_path: Path,
    slide_metadata: list[tuple[int, float, float, Path]],
    transcript_segments: list[Segment],
    output_path: Path,
    model: str = "",
    ocr_pipeline: OcrPipeline | None = None,
    source_url: str | None = None,
    split_slides: bool = False,
    ai_descriptions: dict[str, str] | None = None,
    run_ocr: bool = True,
) -> None:
    """Export slides as Markdown file(s).

    Can be called at any stage:
    - After slide extraction (empty transcript_segments)
    - After transcription (with transcript_segments)

    Args:
        video_path: Path to the source video file.
        slide_metadata: List of (index, start, end, image_path) tuples.
        transcript_segments: Transcript segments from Whisper (can be empty list).
        output_path: Path for the output markdown file.
        model: Whisper model name used for transcription (empty if no transcription yet).
        ocr_pipeline: Optional OCR pipeline for text extraction.
        source_url: Optional source URL for the video.
        split_slides: If True, create separate files (index.md + slide_NNN.md).
                     If False (default), create single slides.md file.
        ai_descriptions: Optional dict mapping slide_id to AI description.
        run_ocr: Whether to run OCR while building this export.
    """
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build default OCR pipeline if none provided
    if run_ocr and ocr_pipeline is None:
        from slidegeist.ocr import build_default_ocr_pipeline

        ocr_pipeline = build_default_ocr_pipeline()

    # Read existing markdown to preserve/merge content
    existing_data = _parse_existing_markdown(output_path)
    has_existing = len(existing_data) > 0

    if has_existing:
        logger.info("Merging with existing slides markdown (%d slides)", len(existing_data))
    else:
        logger.info("Creating slides markdown with %d slides", len(slide_metadata))

    # Process all slides and collect data
    from tqdm import tqdm  # type: ignore[import-untyped]

    slide_sections: list[str] = []
    index_lines: list[str] = []
    total_slides = len(slide_metadata)

    for index, (slide_index, t_start, t_end, image_path) in enumerate(
        tqdm(
            slide_metadata,
            desc="Processing slides",
            unit="slide",
            disable=not run_ocr or not ocr_pipeline,
        )
    ):
        slide_id = image_path.stem or f"slide_{slide_index:03d}"
        image_filename = image_path.name

        # Get existing content for this slide (if any)
        existing_slide = existing_data.get(slide_id, {})

        # Transcript: use new if provided, else keep existing
        if transcript_segments:
            transcript_text = _collect_transcript_text(transcript_segments, t_start, t_end)
        else:
            transcript_text = existing_slide.get("transcript", "")

        # OCR: run if available, else keep existing
        ocr_available = (
            run_ocr
            and ocr_pipeline is not None
            and ocr_pipeline._primary is not None  # type: ignore[union-attr]
            and ocr_pipeline._primary.is_available  # type: ignore[union-attr]
        )
        if ocr_available:
            try:
                transcript_payload = _collect_transcript_payload(
                    transcript_segments, t_start, t_end
                )
                ocr_payload = ocr_pipeline.process(  # type: ignore[union-attr]
                    image_path=image_path,
                    transcript_full_text=transcript_text,
                    transcript_segments=transcript_payload["segments"],
                )
                ocr_text = ocr_payload.get("final_text", "").strip()
                visual_elements = ocr_payload.get("visual_elements", [])
            except Exception as exc:
                logger.warning("OCR failed for %s: %s", image_path, exc)
                ocr_text = existing_slide.get("ocr", "")
                visual_elements = []
        else:
            ocr_text = existing_slide.get("ocr", "")
            visual_elements = []

        if ai_descriptions and slide_id in ai_descriptions:
            ai_description = ai_descriptions[slide_id]
        else:
            ai_description = existing_slide.get("ai_description", "")

        time_str = f"{_format_timestamp(t_start)}-{_format_timestamp(t_end)}"

        if split_slides:
            # Split mode: write individual slide files and create index links
            markdown_content = _build_slide_markdown(
                slide_id=slide_id,
                slide_index=slide_index,
                t_start=t_start,
                t_end=t_end,
                image_filename=image_filename,
                transcript_text=transcript_text,
                ocr_text=ocr_text,
                visual_elements=visual_elements,
                ai_description=ai_description,
            )
            per_slide_path = output_dir / f"{slide_id}.md"
            per_slide_path.write_text(markdown_content, encoding="utf-8")

            index_lines.append(
                f"{slide_index}. [Slide {slide_index}]({slide_id}.md) • "
                f"[![thumb](slides/{image_filename})]({slide_id}.md) • {time_str}"
            )
            logger.debug("Wrote slide %s (%d/%d)", per_slide_path, index + 1, total_slides)
        else:
            # Single file mode: collect sections only (no table of contents)
            section = _build_slide_section(
                slide_index=slide_index,
                t_start=t_start,
                t_end=t_end,
                image_filename=image_filename,
                transcript_text=transcript_text,
                ocr_text=ocr_text,
                visual_elements=visual_elements,
                ai_description=ai_description,
            )
            slide_sections.append(section)

    # Write output file(s)
    if split_slides:
        index_content = _build_index_markdown(
            video_path=video_path,
            source_url=source_url,
            duration=slide_metadata[-1][2] if slide_metadata else 0.0,
            model=model,
            slide_lines=index_lines,
        )
        output_path.write_text(index_content, encoding="utf-8")
        logger.info("Exported slides index to %s", output_path)
    else:
        combined_content = _build_combined_markdown(
            video_path=video_path,
            source_url=source_url,
            duration=slide_metadata[-1][2] if slide_metadata else 0.0,
            model=model,
            slide_sections=slide_sections,
        )
        output_path.write_text(combined_content, encoding="utf-8")
        logger.info("Exported combined slides to %s", output_path)


def _collect_transcript_text(
    transcript_segments: list[Segment],
    start_time: float,
    end_time: float,
) -> str:
    """Collect transcript text overlapping the slide interval.

    With zero or one speaker in the window the output is plain joined text,
    exactly as before speaker labels existed. Only genuine multi-speaker
    windows are broken into ``**SPEAKER_00:** ...`` runs, so single-lecturer
    material - the overwhelming majority - stays byte-identical.
    """
    collected: list[tuple[str | None, str]] = []
    for segment in transcript_segments:
        seg_start = segment["start"]
        seg_end = segment["end"]
        overlap = seg_start < end_time and seg_end > start_time
        if overlap:
            text = segment["text"].strip()
            if text:
                collected.append((segment.get("speaker"), text))

    if not collected:
        return ""

    speakers = {speaker for speaker, _ in collected if speaker}
    if len(speakers) < 2:
        return " ".join(text for _, text in collected)

    lines: list[str] = []
    current_speaker: str | None = None
    current: list[str] = []
    for speaker, text in collected:
        if speaker != current_speaker and current:
            lines.append(_format_speaker_run(current_speaker, current))
            current = []
        current_speaker = speaker
        current.append(text)
    if current:
        lines.append(_format_speaker_run(current_speaker, current))
    return "\n\n".join(lines)


def _format_speaker_run(speaker: str | None, texts: list[str]) -> str:
    """Render one speaker's consecutive text.

    The renderer must never emit a bare ``---`` or a ``#``-prefixed line: both
    terminate a transcript section in the markdown parsers.
    """
    body = " ".join(texts).replace("\n", " ").strip()
    if not speaker:
        return body
    return f"**{speaker}:** {body}"


def _collect_transcript_payload(
    transcript_segments: list[Segment],
    start_time: float,
    end_time: float,
) -> dict[str, Any]:
    """Filter transcript segments to those overlapping the slide interval."""
    segments: list[dict[str, Any]] = []

    for segment in transcript_segments:
        seg_start = segment["start"]
        seg_end = segment["end"]
        overlap = seg_start < end_time and seg_end > start_time

        if not overlap:
            continue

        text = segment["text"].strip()
        if not text:
            continue

        segments.append(
            {
                "start": seg_start,
                "end": seg_end,
                "text": text,
                "words": segment.get("words", []),
            }
        )

    full_text = " ".join(item["text"] for item in segments)

    return {
        "full_text": full_text,
        "segments": segments,
    }


def _format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def _build_slide_section(
    slide_index: int,
    t_start: float,
    t_end: float,
    image_filename: str,
    transcript_text: str,
    ocr_text: str,
    visual_elements: list[str],
    ai_description: str = "",
) -> str:
    """Build Markdown section for a slide in combined mode."""
    slide_id = f"slide_{slide_index:03d}"
    lines = [
        f'<a name="{slide_id}"></a>',
        f"## Slide {slide_index}",
        "",
        f"**Time:** {_format_timestamp(t_start)} - {_format_timestamp(t_end)}",
        "",
        f"[![Slide](slides/{image_filename})](slides/{image_filename})",
        "",
    ]

    if transcript_text:
        lines.extend(
            [
                "### Transcript",
                "",
                transcript_text,
                "",
            ]
        )

    if ocr_text:
        lines.extend(
            [
                "### OCR Text",
                "",
                ocr_text,
                "",
            ]
        )

    if visual_elements:
        elements_str = ", ".join(visual_elements)
        lines.extend(
            [
                f"**Visual Elements:** {elements_str}",
                "",
            ]
        )

    if ai_description:
        lines.extend(
            [
                "### AI Description (for reconstruction)",
                "",
                ai_description,
                "",
            ]
        )

    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def _build_slide_markdown(
    slide_id: str,
    slide_index: int,
    t_start: float,
    t_end: float,
    image_filename: str,
    transcript_text: str,
    ocr_text: str,
    visual_elements: list[str],
    ai_description: str = "",
) -> str:
    """Build Markdown content for a single slide in split mode."""
    lines = [
        "---",
        f"id: {slide_id}",
        f"index: {slide_index}",
        f"time_start: {t_start}",
        f"time_end: {t_end}",
        f"image: slides/{image_filename}",
        "---",
        "",
        f"# Slide {slide_index}",
        "",
        f"[![Slide Image](slides/{image_filename})](slides/{image_filename})",
        "",
    ]

    if transcript_text:
        lines.extend(
            [
                "## Transcript",
                "",
                transcript_text,
                "",
            ]
        )

    if ocr_text:
        lines.extend(
            [
                "## OCR Text",
                "",
                ocr_text,
                "",
            ]
        )

    if visual_elements:
        elements_str = ", ".join(visual_elements)
        lines.extend(
            [
                f"**Visual Elements:** {elements_str}",
                "",
            ]
        )

    if ai_description:
        lines.extend(
            [
                "## AI Description (for reconstruction)",
                "",
                ai_description,
                "",
            ]
        )

    return "\n".join(lines)


def _build_combined_markdown(
    video_path: Path,
    source_url: str | None,
    duration: float,
    model: str,
    slide_sections: list[str],
) -> str:
    """Build combined markdown file with header, index, and all slides."""
    lines = [
        "# Lecture Slides",
        "",
        f"**Video:** {video_path.name}  ",
    ]

    if source_url:
        lines.append(f"**Source:** {source_url}  ")

    duration_str = _format_timestamp(duration)
    processed_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines.extend(
        [
            f"**Duration:** {duration_str}  ",
            f"**Transcription Model:** {model}  ",
            f"**Processed:** {processed_at}",
            "",
            "---",
            "",
        ]
    )

    lines.extend(slide_sections)

    return "\n".join(lines)


def run_ai_descriptions(
    slide_metadata: list[tuple[int, float, float, Path]],
    transcript_segments: list[Segment],
    describer: BaseSlideDescriber,
    ocr_pipeline: OcrPipeline | None = None,
    output_path: Path | None = None,
    force_redo: bool = False,
) -> dict[str, str]:
    """Run AI descriptions on all slides with incremental saving.

    Args:
        slide_metadata: List of (index, start, end, image_path) tuples.
        transcript_segments: Transcript segments for context.
        describer: AI describer instance.
        ocr_pipeline: OCR pipeline to extract text (optional).
        output_path: Path to slides.md for incremental updates (optional).
        force_redo: If True, regenerate all AI descriptions even if they exist.

    Returns:
        Dictionary mapping slide_id to AI description.
    """
    from tqdm import tqdm

    descriptions: dict[str, str] = {}

    from slidegeist.frame_filter import is_complete_frame_description

    # Load existing classified descriptions to skip already processed slides.
    # Legacy or malformed responses are regenerated instead of silently accepted.
    existing_data = {}
    total_slides = len(slide_metadata)

    if output_path and output_path.exists() and not force_redo:
        existing_data = _parse_existing_markdown(output_path)
        # Count how many already have AI descriptions
        existing_count = sum(
            1
            for data in existing_data.values()
            if is_complete_frame_description(data.get("ai_description", ""))
        )
        if existing_count > 0:
            if existing_count == total_slides:
                logger.info(f"All {total_slides} slides have AI descriptions, will skip generation")
            else:
                logger.info(
                    f"Found {existing_count}/{total_slides} existing AI descriptions, will skip those and generate the rest"
                )
    elif force_redo:
        logger.info("Force redo enabled: regenerating ALL AI descriptions")

    for slide_index, t_start, t_end, image_path in tqdm(
        slide_metadata, desc="Generating AI descriptions", unit="slide"
    ):
        slide_id = image_path.stem or f"slide_{slide_index:03d}"

        # Skip if already has AI description (unless force_redo)
        if (
            not force_redo
            and slide_id in existing_data
            and is_complete_frame_description(
                existing_data[slide_id].get("ai_description", "")
            )
        ):
            descriptions[slide_id] = existing_data[slide_id]["ai_description"]
            logger.debug(f"Skipping {slide_id} (already has AI description)")
            continue

        transcript_text = _collect_transcript_text(transcript_segments, t_start, t_end)

        ocr_text = existing_data.get(slide_id, {}).get("ocr", "")
        if not ocr_text and ocr_pipeline is not None:
            try:
                ocr_result = ocr_pipeline.process(image_path, transcript_text, [])
                ocr_text = ocr_result.get("raw_text", "")
            except Exception as exc:
                logger.debug(f"OCR extraction failed for {slide_id}: {exc}")

        try:
            description = describer.describe(image_path, transcript_text, ocr_text)
            if description:
                descriptions[slide_id] = description
                logger.info(f"Generated AI description for {slide_id}")

                # Immediately save to disk for crash recovery
                if output_path:
                    _save_incremental_ai_description(
                        output_path, slide_id, description, existing_data
                    )
            else:
                logger.warning(f"Empty AI description for {slide_id}")
        except Exception as exc:
            logger.error(f"AI description failed for {slide_id}: {exc}")
            raise

    return descriptions


def _save_incremental_ai_description(
    output_path: Path, slide_id: str, ai_description: str, existing_data: dict[str, dict[str, str]]
) -> None:
    """Update slides.md with new AI description for a single slide.

    Args:
        output_path: Path to slides.md file.
        slide_id: Slide identifier (e.g., "slide_001").
        ai_description: Generated AI description text.
        existing_data: Parsed existing markdown data.
    """
    if not output_path.exists():
        logger.warning(f"Cannot save incremental update: {output_path} does not exist")
        return

    try:
        if output_path.name == "index.md":
            slide_path = output_path.parent / f"{slide_id}.md"
            if not slide_path.exists():
                logger.warning(f"Cannot find split slide {slide_id} in {output_path.parent}")
                return
            content = slide_path.read_text(encoding="utf-8")
            heading = "## AI Description (for reconstruction)"
            start = content.find(heading)
            if start >= 0:
                following = re.search(r"^## ", content[start + len(heading) :], re.MULTILINE)
                end = start + len(heading) + following.start() if following else len(content)
                before = content[:start].rstrip()
                after = content[end:].lstrip()
                content = before + (f"\n\n{after}" if after else "")
            content = content.rstrip() + f"\n\n{heading}\n\n{ai_description}\n"
            _atomic_write_text(slide_path, content)
            return

        content = output_path.read_text(encoding="utf-8")
        lines = content.split("\n")

        # Find the slide section
        slide_anchor = f'<a name="{slide_id}"></a>'
        slide_start = None
        for i, line in enumerate(lines):
            if slide_anchor in line:
                slide_start = i
                break

        if slide_start is None:
            logger.warning(f"Could not find slide {slide_id} in {output_path}")
            return

        # Find the next slide or end of file
        next_slide_start = len(lines)
        for i in range(slide_start + 1, len(lines)):
            if lines[i].strip().startswith('<a name="slide_'):
                next_slide_start = i
                break

        # Remove every stale checkpoint section. An interrupted retry can leave
        # more than one, including sections accidentally appended after the
        # slide separator.
        slide_lines = lines[slide_start:next_slide_start]
        cleaned: list[str] = []
        position = 0
        heading = "### AI Description (for reconstruction)"
        while position < len(slide_lines):
            if slide_lines[position].strip() != heading:
                cleaned.append(slide_lines[position])
                position += 1
                continue
            position += 1
            while position < len(slide_lines):
                marker = slide_lines[position].strip()
                if marker.startswith("###") or marker == "---":
                    break
                position += 1

        separator_line = next(
            (index for index in range(len(cleaned) - 1, -1, -1) if cleaned[index].strip() == "---"),
            len(cleaned),
        )
        new_section = [
            heading,
            "",
            ai_description,
            "",
        ]
        cleaned[separator_line:separator_line] = new_section
        lines[slide_start:next_slide_start] = cleaned

        # Write back
        _atomic_write_text(output_path, "\n".join(lines))
        logger.debug(f"Saved AI description for {slide_id} to {output_path}")

    except Exception as exc:
        logger.warning(f"Failed to save incremental AI description for {slide_id}: {exc}")


def _build_index_markdown(
    video_path: Path,
    source_url: str | None,
    duration: float,
    model: str,
    slide_lines: list[str],
) -> str:
    """Build the index Markdown file for split mode."""
    lines = [
        "# Lecture Slides",
        "",
        f"**Video:** {video_path.name}  ",
    ]

    if source_url:
        lines.append(f"**Source:** {source_url}  ")

    duration_str = _format_timestamp(duration)
    processed_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines.extend(
        [
            f"**Duration:** {duration_str}  ",
            f"**Transcription Model:** {model}  ",
            f"**Processed:** {processed_at}",
            "",
            "## Slides",
            "",
        ]
    )

    lines.extend(slide_lines)
    lines.append("")

    return "\n".join(lines)
