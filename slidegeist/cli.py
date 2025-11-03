"""Command-line interface for Slidegeist."""

import argparse
import logging
import sys
from pathlib import Path

from slidegeist import __version__
from slidegeist.constants import (
    DEFAULT_DEVICE,
    DEFAULT_IMAGE_FORMAT,
    DEFAULT_MIN_SCENE_LEN,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SCENE_THRESHOLD,
    DEFAULT_START_OFFSET,
    DEFAULT_WHISPER_MODEL,
)
from slidegeist.download import BrowserType, download_video, get_video_filename, is_url
from slidegeist.ffmpeg import check_ffmpeg_available
from slidegeist.pipeline import process_slides_only, process_video

logger = logging.getLogger(__name__)


def resolve_video_path(
    input_str: str,
    output_dir: Path,
    cookies_from_browser: BrowserType | None = None
) -> Path:
    """Resolve input to video path, downloading if URL.

    Args:
        input_str: Video file path or URL.
        output_dir: Output directory to store downloaded video.
        cookies_from_browser: Browser to extract cookies from for authenticated downloads.

    Returns:
        Path to the video file (local or downloaded).
    """
    if is_url(input_str):
        logger.info(f"Detected URL input: {input_str}")
        return download_video(input_str, output_dir=output_dir, cookies_from_browser=cookies_from_browser)
    return Path(input_str)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging output.

    Args:
        verbose: If True, set log level to DEBUG, otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )


def display_legal_notice() -> None:
    """Display legal notice for tool usage."""
    print("\n" + "=" * 70)
    print("Legal Notice")
    print("-" * 70)
    print("Slidegeist is provided for educational and research purposes only.")
    print("Users must ensure they have the legal right to access, download, or")
    print("process any video files they use with this tool.")
    print("The author does not endorse or facilitate copyright infringement or")
    print("violation of platform terms of service.")
    print("=" * 70 + "\n")


def check_prerequisites() -> None:
    """Check that required external tools are available.

    Raises:
        SystemExit: If FFmpeg is not found.
    """
    if not check_ffmpeg_available():
        logger.error("FFmpeg not found in PATH")
        logger.error("Please install FFmpeg:")
        logger.error("  macOS:        brew install ffmpeg")
        logger.error("  Ubuntu/Debian: sudo apt-get install ffmpeg")
        logger.error("  Windows:      winget install ffmpeg")
        sys.exit(1)


def validate_scene_threshold(threshold: float) -> None:
    """Validate scene detection threshold is in valid range.

    Args:
        threshold: Scene detection threshold value.

    Raises:
        SystemExit: If threshold is out of valid range (0.0-1.0).
    """
    if not 0.0 <= threshold <= 1.0:
        logger.error(f"Invalid scene threshold: {threshold}")
        logger.error("Scene threshold must be between 0.0 and 1.0")
        logger.error("  Lower values = more sensitive (detect subtle changes)")
        logger.error("  Higher values = less sensitive (only major transitions)")
        logger.error(f"  Recommended range: 0.015-0.05 (default: {DEFAULT_SCENE_THRESHOLD})")
        sys.exit(1)


def handle_process(args: argparse.Namespace) -> None:
    """Handle 'slidegeist process' command."""
    try:
        display_legal_notice()
        check_prerequisites()
        validate_scene_threshold(args.scene_threshold)

        # Determine output directory early (before download)
        output_dir = args.out
        if output_dir == Path(DEFAULT_OUTPUT_DIR):
            # Need to determine from input
            if is_url(args.input):
                # For URLs, get video filename before downloading
                video_filename = get_video_filename(args.input, getattr(args, 'cookies_from_browser', None))
                output_dir = Path.cwd() / video_filename
            else:
                # For local files, use video filename immediately
                output_dir = Path.cwd() / Path(args.input).stem

        video_path = resolve_video_path(
            args.input, output_dir, getattr(args, 'cookies_from_browser', None)
        )

        source_url = args.input if args.input.startswith(('http://', 'https://')) else None

        result = process_video(
            video_path=video_path,
            output_dir=output_dir,
            scene_threshold=args.scene_threshold,
            min_scene_len=args.min_scene_len,
            start_offset=args.start_offset,
            model=args.model,
            source_url=source_url,
            device=args.device,
            image_format=getattr(args, 'format', DEFAULT_IMAGE_FORMAT),
            split_slides=getattr(args, 'split', False),
            retry_failed=getattr(args, 'retry_failed', False),
            force_redo_ai=getattr(args, 'force_redo_ai', False)
        )

        print("\n" + "=" * 60)
        print("✓ Processing complete!")
        print("=" * 60)
        if 'slides' in result:
            print(f"  Slides:      {len(result['slides'])} images")  # type: ignore
        if 'slides_md' in result:
            print(f"  Markdown:    {result['slides_md']}")
        print(f"  Output dir:  {result['output_dir']}")
        print("=" * 60)

    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        sys.exit(1)


def handle_slides(args: argparse.Namespace) -> None:
    """Handle 'slidegeist slides' command."""
    try:
        display_legal_notice()
        check_prerequisites()
        validate_scene_threshold(args.scene_threshold)

        # Determine output directory early (before download)
        output_dir = args.out
        if output_dir == Path(DEFAULT_OUTPUT_DIR):
            if is_url(args.input):
                # For URLs, get video filename before downloading
                video_filename = get_video_filename(args.input, getattr(args, 'cookies_from_browser', None))
                output_dir = Path.cwd() / video_filename
            else:
                output_dir = Path.cwd() / Path(args.input).stem

        video_path = resolve_video_path(
            args.input, output_dir, getattr(args, 'cookies_from_browser', None)
        )

        result = process_slides_only(
            video_path=video_path,
            output_dir=output_dir,
            scene_threshold=args.scene_threshold,
            min_scene_len=args.min_scene_len,
            start_offset=args.start_offset,
            image_format=args.format
        )

        print(f"\n✓ Extracted {len(result['slides'])} slides")
        print(f"  Output dir: {result['output_dir']}")

    except Exception as e:
        logger.error(f"Slide extraction failed: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        sys.exit(1)


def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="slidegeist",
        description="Extract slides and transcripts from lecture videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process video (slides + transcript with large-v3 model) - default mode
  slidegeist lecture.mp4

  # Use smaller/faster model
  slidegeist lecture.mp4 --model tiny

  # Use GPU explicitly
  slidegeist lecture.mp4 --device cuda

  # Extract only slides (no transcription)
  slidegeist slides lecture.mp4

  # Explicit process command (same as default)
  slidegeist process lecture.mp4
        """
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    # Create parent parsers for common arguments (for subcommands)
    verbose_parent = argparse.ArgumentParser(add_help=False)
    verbose_parent.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    common_parent = argparse.ArgumentParser(add_help=False, parents=[verbose_parent])
    common_parent.add_argument(
        "input",
        type=str,
        help="Input video file path or URL (YouTube, Mediasite, etc.)"
    )
    common_parent.add_argument(
        "--out",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help="Output directory (default: video filename)"
    )
    common_parent.add_argument(
        "--cookies-from-browser",
        choices=["firefox", "safari", "chrome", "chromium", "edge", "opera", "brave"],
        help="Browser to extract cookies from for authenticated video downloads"
    )

    slides_parent = argparse.ArgumentParser(add_help=False)
    slides_parent.add_argument(
        "--scene-threshold",
        type=float,
        default=DEFAULT_SCENE_THRESHOLD,
        metavar="NUM",
        help=f"Scene detection sensitivity 0.0-1.0, lower=more sensitive (default: {DEFAULT_SCENE_THRESHOLD})"
    )
    slides_parent.add_argument(
        "--min-scene-len",
        type=float,
        default=DEFAULT_MIN_SCENE_LEN,
        metavar="SEC",
        help=f"Minimum scene length in seconds (default: {DEFAULT_MIN_SCENE_LEN})"
    )
    slides_parent.add_argument(
        "--start-offset",
        type=float,
        default=DEFAULT_START_OFFSET,
        metavar="SEC",
        help=f"Skip first N seconds to avoid mouse movement (default: {DEFAULT_START_OFFSET})"
    )
    slides_parent.add_argument(
        "--format",
        default=DEFAULT_IMAGE_FORMAT,
        choices=["jpg", "png"],
        help=f"Slide image format (default: {DEFAULT_IMAGE_FORMAT})"
    )

    subparsers = parser.add_subparsers(dest="command", required=False)

    # Process command (full pipeline)
    process_parser = subparsers.add_parser(
        "process",
        parents=[common_parent, slides_parent],
        help="Process video (extract slides and transcript)"
    )
    process_parser.add_argument(
        "--model",
        default=DEFAULT_WHISPER_MODEL,
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help=f"Whisper model size (default: {DEFAULT_WHISPER_MODEL})"
    )
    process_parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        choices=["cpu", "cuda", "auto"],
        help=f"Processing device (default: {DEFAULT_DEVICE} - uses MLX on Apple Silicon if available)"
    )
    process_parser.add_argument(
        "--split",
        action="store_true",
        help="Create separate markdown files (index.md + slide_NNN.md) instead of single slides.md"
    )
    process_parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry previously failed stages (transcription, OCR, AI descriptions)"
    )
    process_parser.add_argument(
        "--force-redo-ai",
        action="store_true",
        help="Regenerate ALL AI descriptions even if they already exist"
    )

    # Slides command
    subparsers.add_parser(
        "slides",
        parents=[common_parent, slides_parent],
        help="Extract only slides (no transcription)"
    )

    # Parse arguments with special handling for default mode
    # Check if first positional arg is a subcommand or a video input
    import sys as sys_module
    argv = sys_module.argv[1:]

    # If first arg is not a known subcommand and looks like input, inject 'process'
    if argv and argv[0] not in ['-h', '--help', '--version', '-v', '--verbose'] and \
       argv[0] not in ['process', 'slides'] and \
       not argv[0].startswith('-'):
        # First positional arg is likely video input, default to process mode
        argv = ['process'] + argv
        args = parser.parse_args(argv)
    else:
        args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # If no command specified, show help
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Dispatch to subcommand handlers
    if args.command == "process":
        handle_process(args)
    elif args.command == "slides":
        handle_slides(args)


if __name__ == "__main__":
    main()
