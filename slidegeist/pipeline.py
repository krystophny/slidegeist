"""Main processing pipeline orchestration."""

import logging
from pathlib import Path

from slidegeist.constants import (
    DEFAULT_DEVICE,
    DEFAULT_IMAGE_FORMAT,
    DEFAULT_MIN_SCENE_LEN,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SCENE_THRESHOLD,
    DEFAULT_START_OFFSET,
    DEFAULT_WHISPER_MODEL,
)
from slidegeist.export import export_slides_json
from slidegeist.ffmpeg import detect_scenes
from slidegeist.ocr import OcrPipeline
from slidegeist.slides import extract_slides
from slidegeist.transcribe import transcribe_video

logger = logging.getLogger(__name__)


def has_existing_slides(output_dir: Path) -> bool:
    """Check if output directory contains extracted slides.

    Args:
        output_dir: Directory to check for slides.

    Returns:
        True if slides subdirectory exists with image files, False otherwise.
    """
    slides_dir = output_dir / "slides"
    if not slides_dir.exists():
        return False

    image_extensions = {".jpg", ".jpeg", ".png"}
    slide_files = [f for f in slides_dir.iterdir() if f.suffix.lower() in image_extensions]
    return len(slide_files) > 0


def find_video_file(output_dir: Path) -> Path | None:
    """Find video file in output directory.

    Args:
        output_dir: Directory to search for video file.

    Returns:
        Path to video file if found, None otherwise.
    """
    if not output_dir.exists():
        return None

    video_extensions = {".mp4", ".mkv", ".webm", ".avi", ".mov"}
    for file in output_dir.iterdir():
        if file.suffix.lower() in video_extensions:
            return file

    return None


def can_resume_from_slides(output_dir: Path) -> bool:
    """Check if processing can resume from existing slides.

    Args:
        output_dir: Directory to check.

    Returns:
        True if directory has both video file and extracted slides, False otherwise.
    """
    return find_video_file(output_dir) is not None and has_existing_slides(output_dir)


def load_existing_slide_metadata(output_dir: Path) -> list[tuple[int, float, float, Path]]:
    """Load metadata for existing slides in output directory.

    Args:
        output_dir: Directory containing slides subdirectory.

    Returns:
        List of tuples (slide_number, start_time, end_time, path) for each slide.
        Times are set to 0.0 as they need to be reconstructed from scene detection
        or transcript data.
    """
    slides_dir = output_dir / "slides"
    if not slides_dir.exists():
        return []

    image_extensions = {".jpg", ".jpeg", ".png"}
    slide_files = sorted(
        [f for f in slides_dir.iterdir() if f.suffix.lower() in image_extensions],
        key=lambda p: p.name,
    )

    metadata: list[tuple[int, float, float, Path]] = []
    for idx, slide_path in enumerate(slide_files, start=1):
        metadata.append((idx, 0.0, 0.0, slide_path))

    return metadata


def detect_completed_stages(output_dir: Path) -> dict[str, bool]:
    """Detect which processing stages have been completed.

    Analyzes slides.md or index.md to determine what's already done.

    Args:
        output_dir: Directory to check.

    Returns:
        Dict with keys: 'slides', 'transcription', 'ocr', 'ai_description'
        Each value is True if that stage is completed.
    """
    stages = {
        "slides": False,
        "transcription": False,
        "ocr": False,
        "ai_description": False,
    }

    # Check for slides directory
    stages["slides"] = has_existing_slides(output_dir)

    # Check markdown files for content
    markdown_path = output_dir / "slides.md"
    if not markdown_path.exists():
        markdown_path = output_dir / "index.md"

    if not markdown_path.exists():
        return stages

    try:
        content = markdown_path.read_text(encoding="utf-8")

        # Detect transcription: look for "### Transcript" sections with content
        if "### Transcript" in content:
            # Check if there's actual transcript content (not just empty sections)
            lines = content.split("\n")
            found_transcript_content = False

            for i, line in enumerate(lines):
                if line.strip() == "### Transcript":
                    # Check next few lines for non-empty content
                    for j in range(i + 1, min(i + 10, len(lines))):
                        next_line = lines[j].strip()
                        # Stop at next section or separator
                        if next_line.startswith("#") or next_line == "---":
                            break
                        # Found actual content
                        if next_line and not next_line.startswith("**"):
                            found_transcript_content = True
                            break
                    if found_transcript_content:
                        break

            stages["transcription"] = found_transcript_content

        # Detect OCR: look for "### OCR Text" or "## OCR Text" sections
        if "OCR Text" in content:
            stages["ocr"] = True

        # Detect AI descriptions: look for "### AI Description" sections
        if "AI Description" in content:
            stages["ai_description"] = True

    except Exception as e:
        logger.debug(f"Could not parse markdown for stage detection: {e}")

    return stages


def process_video(
    video_path: Path,
    output_dir: Path,
    scene_threshold: float = DEFAULT_SCENE_THRESHOLD,
    min_scene_len: float = DEFAULT_MIN_SCENE_LEN,
    start_offset: float = DEFAULT_START_OFFSET,
    model: str = DEFAULT_WHISPER_MODEL,
    source_url: str | None = None,
    device: str = DEFAULT_DEVICE,
    image_format: str = DEFAULT_IMAGE_FORMAT,
    skip_slides: bool = False,
    skip_transcription: bool = False,
    split_slides: bool = False,
    ocr_pipeline: OcrPipeline | None = None,
) -> dict[str, Path | list[Path]]:
    """Process a video and return generated artifacts.

    Returns dictionary always containing ``output_dir`` and optionally:

    * ``slides`` – list of slide image paths when slides are extracted
    * ``slides_json`` – path to slides.json when transcription and slide
      extraction both succeed

    Raises:
        FileNotFoundError: If video file does not exist.
        Exception: For failures in downstream processing stages.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Use video filename (without extension) as default output directory
    if output_dir == Path(DEFAULT_OUTPUT_DIR):
        output_dir = Path.cwd() / video_path.stem

    logger.info(f"Processing video: {video_path}")
    logger.info(f"Output directory: {output_dir}")

    # Detect which stages are already completed
    completed_stages = detect_completed_stages(output_dir)
    logger.info("=" * 60)
    logger.info("STAGE DETECTION")
    logger.info("=" * 60)
    logger.info(f"✓ Slides extracted: {completed_stages['slides']}")
    logger.info(f"✓ Transcription done: {completed_stages['transcription']}")
    logger.info(f"✓ OCR done: {completed_stages['ocr']}")
    logger.info(f"✓ AI descriptions done: {completed_stages['ai_description']}")

    # Check if we can resume from existing work
    resume_from_existing = completed_stages["slides"] and not skip_slides
    if resume_from_existing:
        existing_video = find_video_file(output_dir)
        if existing_video:
            video_path = existing_video
            logger.info(f"Using existing video: {video_path}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Path | list[Path]] = {"output_dir": output_dir}

    # Step 1: Scene detection and slide extraction (or load existing)
    slide_metadata: list[tuple[int, float, float, Path]] = []
    scene_timestamps: list[float] = []

    if completed_stages["slides"] and not skip_slides:
        # Resume: load existing slides without re-extraction
        logger.info("=" * 60)
        logger.info("STEP 1: Loading existing slides")
        logger.info("=" * 60)
        slide_metadata = load_existing_slide_metadata(output_dir)
        results["slides"] = [path for _, _, _, path in slide_metadata]
        logger.info(f"Loaded {len(slide_metadata)} existing slides")
    elif not skip_slides:
        # Normal flow: detect scenes and extract slides
        logger.info("=" * 60)
        logger.info("STEP 1: Scene Detection")
        logger.info("=" * 60)

        scene_timestamps = detect_scenes(
            video_path,
            threshold=scene_threshold,
            min_scene_len=min_scene_len,
            start_offset=start_offset,
        )

        if not scene_timestamps:
            logger.warning("No scene changes detected. Extracting single slide.")

        # Step 2: Extract slides into output directory
        logger.info("=" * 60)
        logger.info("STEP 2: Slide Extraction")
        logger.info("=" * 60)

        slide_metadata = extract_slides(video_path, scene_timestamps, output_dir, image_format)
        results["slides"] = [path for _, _, _, path in slide_metadata]

        # Checkpoint: Save markdown with slides immediately after extraction
        logger.info("Saving slides markdown checkpoint")
        markdown_path = output_dir / ("index.md" if split_slides else "slides.md")
        export_slides_json(
            video_path,
            slide_metadata,
            [],  # No transcript yet
            markdown_path,
            model="",  # No model yet
            ocr_pipeline=ocr_pipeline,
            source_url=source_url,
            split_slides=split_slides,
        )
        results["slides_md"] = markdown_path

    # Step 3: Transcription (skip if already done)
    transcript_segments = []
    transcription_needed = not skip_transcription and not completed_stages["transcription"]

    if transcription_needed:
        logger.info("=" * 60)
        logger.info("STEP 3: Audio Transcription")
        logger.info("=" * 60)

        transcript_data = transcribe_video(video_path, model_size=model, device=device)
        transcript_segments = transcript_data["segments"]
    elif completed_stages["transcription"]:
        logger.info("=" * 60)
        logger.info("STEP 3: Transcription already completed (skipping)")
        logger.info("=" * 60)

    # Step 4: Update/re-run markdown export
    # Always run export if we did any new processing, OR if OCR/AI stages need updating
    has_slides = len(slide_metadata) > 0
    has_new_data = transcription_needed  # Did we just run transcription?
    needs_ocr_update = has_slides and not completed_stages["ocr"]
    needs_ai_update = has_slides and not completed_stages["ai_description"]

    # Re-export if: new transcription, need OCR, or need AI descriptions
    should_export = has_slides and (has_new_data or needs_ocr_update or needs_ai_update)

    if should_export:
        logger.info("=" * 60)
        logger.info("STEP 4: Updating slides markdown")
        logger.info("=" * 60)

        if needs_ocr_update:
            logger.info("Running OCR on slides...")
        if needs_ai_update:
            logger.info("Generating AI descriptions...")

        markdown_path = output_dir / ("index.md" if split_slides else "slides.md")
        export_slides_json(
            video_path,
            slide_metadata,
            transcript_segments,  # Empty if transcription was skipped
            markdown_path,
            model,
            ocr_pipeline=ocr_pipeline,
            source_url=source_url,
            split_slides=split_slides,
        )
        results["slides_md"] = markdown_path

    # Summary
    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    if has_slides:
        action = "Loaded" if completed_stages["slides"] else "Extracted"
        logger.info(f"✓ {action} {len(slide_metadata)} slides")
    if completed_stages["transcription"] or transcription_needed:
        logger.info("✓ Transcription available")
    if completed_stages["ocr"] or needs_ocr_update:
        logger.info("✓ OCR complete")
    if completed_stages["ai_description"] or needs_ai_update:
        logger.info("✓ AI descriptions generated")
    if should_export:
        logger.info("✓ Updated slides markdown")
    logger.info(f"✓ All outputs in: {output_dir}")

    return results


def process_slides_only(
    video_path: Path,
    output_dir: Path,
    scene_threshold: float = DEFAULT_SCENE_THRESHOLD,
    min_scene_len: float = DEFAULT_MIN_SCENE_LEN,
    start_offset: float = DEFAULT_START_OFFSET,
    image_format: str = DEFAULT_IMAGE_FORMAT,
) -> dict:
    """Extract only slides from video (no transcription).

    Args:
        video_path: Path to the input video file.
        output_dir: Directory where slide images will be saved.
        scene_threshold: Scene detection threshold (0-1 scale, lower = more sensitive).
        min_scene_len: Minimum scene length in seconds.
        start_offset: Skip first N seconds to avoid setup noise.
        image_format: Output image format (jpg or png).

    Returns:
        Dictionary containing ``output_dir`` and ``slides`` entries.
    """
    logger.info("Extracting slides only (no transcription)")
    result = process_video(
        video_path,
        output_dir,
        scene_threshold=scene_threshold,
        min_scene_len=min_scene_len,
        start_offset=start_offset,
        image_format=image_format,
        skip_transcription=True,
    )
    return result
