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

    Parses slides.md or index.md to extract timing information.

    Args:
        output_dir: Directory containing slides subdirectory.

    Returns:
        List of tuples (slide_number, start_time, end_time, path) for each slide.
    """
    slides_dir = output_dir / "slides"
    if not slides_dir.exists():
        return []

    image_extensions = {".jpg", ".jpeg", ".png"}
    slide_files = sorted(
        [f for f in slides_dir.iterdir() if f.suffix.lower() in image_extensions],
        key=lambda p: p.name,
    )

    # Try to parse timing information from markdown
    timing_info: dict[str, tuple[float, float]] = {}
    markdown_path = output_dir / "slides.md"
    if not markdown_path.exists():
        markdown_path = output_dir / "index.md"

    if markdown_path.exists():
        try:
            import re
            content = markdown_path.read_text(encoding="utf-8")
            # Look for patterns like: **Time:** 00:05 - 01:23
            # or in YAML frontmatter: time_start: 5.0 / time_end: 83.0

            # Pattern 1: YAML frontmatter (split mode)
            yaml_pattern = r"id:\s*(\S+).*?time_start:\s*([\d.]+).*?time_end:\s*([\d.]+)"
            for match in re.finditer(yaml_pattern, content, re.DOTALL):
                slide_id = match.group(1)
                t_start = float(match.group(2))
                t_end = float(match.group(3))
                timing_info[slide_id] = (t_start, t_end)

            # Pattern 2: **Time:** format (combined mode)
            time_pattern = r'<a name="(slide_\d+)"></a>.*?\*\*Time:\*\*\s*(\d+):(\d+)\s*-\s*(\d+):(\d+)'
            for match in re.finditer(time_pattern, content, re.DOTALL):
                slide_id = match.group(1)
                start_min, start_sec = int(match.group(2)), int(match.group(3))
                end_min, end_sec = int(match.group(4)), int(match.group(5))
                t_start = start_min * 60 + start_sec
                t_end = end_min * 60 + end_sec
                timing_info[slide_id] = (t_start, t_end)

        except Exception as e:
            logger.debug(f"Could not parse timing info from markdown: {e}")

    metadata: list[tuple[int, float, float, Path]] = []
    for idx, slide_path in enumerate(slide_files, start=1):
        slide_id = slide_path.stem
        if slide_id in timing_info:
            t_start, t_end = timing_info[slide_id]
            metadata.append((idx, t_start, t_end, slide_path))
        else:
            # Default to 0.0 if not found
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


def detect_failed_stages(output_dir: Path) -> dict[str, bool]:
    """Detect which processing stages have failed.

    Checks for .failed marker files in the output directory.

    Args:
        output_dir: Directory to check.

    Returns:
        Dict with keys: 'transcription', 'ocr', 'ai_description'
        Each value is True if that stage has failed.
    """
    return {
        "transcription": (output_dir / ".transcription_failed").exists(),
        "ocr": (output_dir / ".ocr_failed").exists(),
        "ai_description": (output_dir / ".ai_description_failed").exists(),
    }


def mark_stage_failed(output_dir: Path, stage: str, error_msg: str) -> None:
    """Mark a stage as failed with an error message.

    Args:
        output_dir: Output directory.
        stage: Stage name ('transcription', 'ocr', 'ai_description').
        error_msg: Error message to store.
    """
    marker_file = output_dir / f".{stage}_failed"
    try:
        marker_file.write_text(error_msg, encoding="utf-8")
        logger.info(f"Marked {stage} as failed: {marker_file}")
    except Exception as e:
        logger.debug(f"Could not write failure marker: {e}")


def clear_stage_failure(output_dir: Path, stage: str) -> None:
    """Clear a stage failure marker.

    Args:
        output_dir: Output directory.
        stage: Stage name ('transcription', 'ocr', 'ai_description').
    """
    marker_file = output_dir / f".{stage}_failed"
    if marker_file.exists():
        try:
            marker_file.unlink()
            logger.debug(f"Cleared failure marker: {marker_file}")
        except Exception as e:
            logger.debug(f"Could not remove failure marker: {e}")


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
    retry_failed: bool = False,
    force_redo_ai: bool = False,
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

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect which stages are already completed and which have failed
    completed_stages = detect_completed_stages(output_dir)
    failed_stages = detect_failed_stages(output_dir)

    logger.info("=" * 60)
    logger.info("STAGE DETECTION")
    logger.info("=" * 60)
    logger.info(f"✓ Slides extracted: {completed_stages['slides']}")
    logger.info(f"✓ Transcription done: {completed_stages['transcription']}")
    logger.info(f"✓ OCR done: {completed_stages['ocr']}")
    logger.info(f"✓ AI descriptions done: {completed_stages['ai_description']}")

    if any(failed_stages.values()):
        logger.info("")
        logger.info("Previous failures detected:")
        if failed_stages["transcription"]:
            logger.warning("✗ Transcription failed previously")
        if failed_stages["ocr"]:
            logger.warning("✗ OCR failed previously")
        if failed_stages["ai_description"]:
            logger.warning("✗ AI description failed previously")
        if retry_failed:
            logger.info("--retry-failed enabled: will retry failed stages")
        else:
            logger.info("Use --retry-failed to retry failed stages")

    # Check if we can resume from existing work
    resume_from_existing = completed_stages["slides"] and not skip_slides
    if resume_from_existing:
        existing_video = find_video_file(output_dir)
        if existing_video:
            video_path = existing_video
            logger.info(f"Using existing video: {video_path}")

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

    # Step 3: Transcription (skip if already done, or failed without retry)
    transcript_segments = []
    should_skip_transcription = (
        skip_transcription
        or completed_stages["transcription"]  # Always skip if completed
        or (failed_stages["transcription"] and not retry_failed)  # Skip failed unless retrying
    )

    if not should_skip_transcription:
        logger.info("=" * 60)
        logger.info("STEP 3: Audio Transcription")
        logger.info("=" * 60)

        try:
            transcript_data = transcribe_video(video_path, model_size=model, device=device)
            transcript_segments = transcript_data["segments"]
            clear_stage_failure(output_dir, "transcription")
        except Exception as exc:
            error_msg = f"Transcription failed: {exc}\n\nTo fix:\n"
            error_msg += "1. Install openai-whisper: pip install openai-whisper\n"
            error_msg += "2. For MLX (Apple Silicon): pip install mlx-whisper\n"
            error_msg += "3. For CUDA: Install PyTorch with CUDA support first\n"
            logger.error(error_msg)
            mark_stage_failed(output_dir, "transcription", error_msg)
            # Continue without transcription
    elif completed_stages["transcription"]:
        logger.info("=" * 60)
        logger.info("STEP 3: Transcription already completed (skipping)")
        logger.info("=" * 60)
    elif failed_stages["transcription"]:
        logger.info("=" * 60)
        logger.info("STEP 3: Transcription failed previously (skipping)")
        logger.info("=" * 60)
        # Read failure message if available
        failure_file = output_dir / ".transcription_failed"
        if failure_file.exists():
            try:
                failure_msg = failure_file.read_text(encoding="utf-8")
                logger.warning(failure_msg)
            except Exception:
                pass

    # Step 4: OCR on slides
    has_slides = len(slide_metadata) > 0
    transcription_just_ran = not should_skip_transcription and len(transcript_segments) > 0
    needs_ocr = has_slides and not completed_stages["ocr"]

    if needs_ocr:
        logger.info("=" * 60)
        logger.info("STEP 4: OCR Text Extraction")
        logger.info("=" * 60)

        markdown_path = output_dir / ("index.md" if split_slides else "slides.md")
        export_slides_json(
            video_path,
            slide_metadata,
            transcript_segments,
            markdown_path,
            model,
            ocr_pipeline=ocr_pipeline,
            source_url=source_url,
            split_slides=split_slides,
        )
        results["slides_md"] = markdown_path
    elif transcription_just_ran and has_slides:
        logger.info("=" * 60)
        logger.info("STEP 4: Updating markdown with transcript")
        logger.info("=" * 60)

        markdown_path = output_dir / ("index.md" if split_slides else "slides.md")
        export_slides_json(
            video_path,
            slide_metadata,
            transcript_segments,
            markdown_path,
            model,
            ocr_pipeline=ocr_pipeline,
            source_url=source_url,
            split_slides=split_slides,
        )
        results["slides_md"] = markdown_path

    # Step 5: AI Descriptions
    should_skip_ai = (
        (completed_stages["ai_description"] and not retry_failed)
        or (failed_stages["ai_description"] and not retry_failed)
    )
    needs_ai = has_slides and not completed_stages["ai_description"]

    if needs_ai and not should_skip_ai:
        # Free GPU memory before loading vision model
        logger.info("Freeing GPU memory before AI descriptions...")
        try:
            import gc
            import torch  # type: ignore[import-untyped]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU memory cleared")
        except ImportError:
            pass  # PyTorch not installed, skip cleanup
        except Exception as e:
            logger.debug(f"GPU cleanup warning: {e}")

        logger.info("=" * 60)
        logger.info("STEP 5: AI Slide Descriptions")
        logger.info("=" * 60)

        try:
            from slidegeist.ai_description import build_ai_describer
            from slidegeist.export import run_ai_descriptions

            describer = build_ai_describer()
            if describer is None:
                error_msg = "AI describer not available\n\nTo fix:\n"
                error_msg += "1. For MLX (Apple Silicon): pip install mlx-vlm\n"
                error_msg += "2. For PyTorch: pip install torch transformers\n"
                error_msg += "3. Install Qwen3-VL model (will download on first run)\n"
                logger.error(error_msg)
                mark_stage_failed(output_dir, "ai_description", error_msg)
            else:
                logger.info(f"Using {describer.name} for AI descriptions")
                markdown_path = output_dir / ("index.md" if split_slides else "slides.md")

                ai_descriptions = run_ai_descriptions(
                    slide_metadata,
                    transcript_segments,
                    describer,
                    ocr_pipeline,
                    output_path=markdown_path,
                    force_redo=force_redo_ai,
                )
                export_slides_json(
                    video_path,
                    slide_metadata,
                    transcript_segments,
                    markdown_path,
                    model,
                    ocr_pipeline=ocr_pipeline,
                    source_url=source_url,
                    split_slides=split_slides,
                    ai_descriptions=ai_descriptions,
                )
                results["slides_md"] = markdown_path
                clear_stage_failure(output_dir, "ai_description")

        except Exception as exc:
            error_msg = f"AI description failed: {exc}\n\nTo fix:\n"
            error_msg += "1. For MLX (Apple Silicon): pip install mlx-vlm\n"
            error_msg += "2. For PyTorch CUDA: pip install torch transformers torchvision\n"
            error_msg += "3. For PyTorch CPU: pip install torch transformers torchvision\n"
            logger.error("=" * 60)
            logger.error("AI DESCRIPTION FAILED")
            logger.error("=" * 60)
            logger.error(error_msg)
            logger.error("=" * 60)
            mark_stage_failed(output_dir, "ai_description", error_msg)

    elif completed_stages["ai_description"]:
        logger.info("=" * 60)
        logger.info("STEP 5: AI descriptions already completed (skipping)")
        logger.info("=" * 60)
    elif failed_stages["ai_description"]:
        logger.info("=" * 60)
        logger.info("STEP 5: AI description failed previously (skipping)")
        logger.info("=" * 60)
        failure_file = output_dir / ".ai_description_failed"
        if failure_file.exists():
            try:
                failure_msg = failure_file.read_text(encoding="utf-8")
                logger.warning(failure_msg)
            except Exception:
                pass

    # Summary
    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    if has_slides:
        action = "Loaded" if completed_stages["slides"] else "Extracted"
        logger.info(f"✓ {action} {len(slide_metadata)} slides")
    if completed_stages["transcription"] or transcription_just_ran:
        logger.info("✓ Transcription available")
    if completed_stages["ocr"] or needs_ocr:
        logger.info("✓ OCR complete")
    if completed_stages["ai_description"]:
        logger.info("✓ AI descriptions generated")
    elif failed_stages.get("ai_description"):
        logger.warning("✗ AI descriptions FAILED - check .ai_description_failed for details")
    if results.get("slides_md"):
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
