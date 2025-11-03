# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Slidegeist extracts slides and timestamped transcripts from lecture videos using FFmpeg scene detection and Whisper transcription. The project emphasizes minimal dependencies, research-based methods, and platform-optimized performance (MLX on Apple Silicon).

## Build, Test, and Lint Commands

### Development Setup
```bash
pip install -e ".[dev]"          # Install in editable mode with dev dependencies
pip install -e ".[mlx]"          # Add MLX support for Apple Silicon
```

### Testing
```bash
pytest                           # Run all tests (excludes manual tests)
pytest -v                        # Verbose output
pytest tests/test_export.py      # Run specific test file
pytest -m "not slow"             # Skip slow tests
pytest --cov=slidegeist --cov-report=html  # Coverage report
```

### Code Quality
```bash
ruff check slidegeist/           # Run linter
ruff check --fix slidegeist/     # Auto-fix linting issues
ruff format slidegeist/          # Auto-format code
mypy slidegeist/                 # Type check entire package
mypy slidegeist/file.py --strict # Strict type checking on single file
```

### Running the CLI
```bash
slidegeist video.mp4                        # Process video (default: slides + transcript)
slidegeist video.mp4 --out output/          # Specify output directory
slidegeist video.mp4 --scene-threshold 0.02 # Adjust scene detection sensitivity
slidegeist video.mp4 --model base           # Use faster Whisper model
slidegeist slides video.mp4                 # Extract only slides (no transcription)
```

## Architecture

### Processing Pipeline (pipeline.py)

The main `process_video()` function orchestrates processing with smart resume capabilities:

**Smart Resume**: If output directory contains both a video file and a slides subdirectory with images, automatically skips slide extraction and resumes from transcription. This enables re-running slidegeist with the same URL to add transcription to existing slides.

**Processing Steps**:

1. **Scene Detection** (ffmpeg_scene.py): Uses FFmpeg's SAD-based scene filter with Opencast-style optimization
   - Iteratively adjusts threshold to target ~30 segments/hour (typical lecture pace)
   - Merges segments shorter than 2 seconds to filter out rapid flickers
   - `--scene-threshold` serves as the optimizer's starting point

2. **Slide Extraction** (slides.py): Extracts frames at 80% through each detected segment
   - Simple numbered filenames: slide_001.jpg, slide_002.jpg, etc.
   - Supports JPG and PNG formats

3. **Transcription** (transcribe.py): Uses Whisper for speech-to-text with word-level timestamps
   - Auto-detects MLX on Apple Silicon for 2-3x speedup
   - Falls back to faster-whisper on other platforms
   - VAD (Voice Activity Detection) filtering enabled by default

4. **OCR** (ocr.py): Optional Tesseract OCR with Qwen3-VL vision model refinement
   - Two-stage pipeline: fast Tesseract baseline, then optional MLX-based enhancement
   - Qwen3-VL only available on Apple Silicon with MLX

5. **Export** (export.py): Generates Markdown files with YAML front matter
   - Default: Single `slides.md` with table of contents (LLM-friendly)
   - Split mode (`--split`): Separate files per slide with `index.md`

### Key Design Decisions

- **Opencast compatibility**: Scene detection threshold and optimization mirror Opencast's VideoSegmenterService implementation
- **Platform optimization**: MLX auto-detection for Apple Silicon; faster-whisper for other platforms
- **Research-based defaults**: Scene threshold (0.025), target segments/hour (30), minimum segment length (2s) are based on Opencast research
- **Minimal dependencies**: Core functionality works with just FFmpeg, faster-whisper, opencv-python, and pytesseract

### Scene Detection Implementation

Two complementary implementations exist:

1. **ffmpeg_scene.py** (default): FFmpeg's built-in scene filter with Opencast optimizer
   - Fast, battle-tested, used in production
   - SAD (Sum of Absolute Differences) metric
   - Includes `detect_scenes_ffmpeg()` and `merge_short_segments()`

2. **pixel_diff_detector.py** (research/experimental): Custom implementation for analysis
   - Supports multiple methods: SAD, z-score (rolling window), histogram
   - Used by `scripts/plot_threshold_sweep.py` for research and tuning
   - Not used in main CLI pipeline

### Device and Model Selection

- Device selection logic in transcribe.py:
  - `auto`: Auto-detects best device in priority order: MLX (Apple Silicon) → CUDA (NVIDIA GPU) → CPU
  - `cuda`: Explicit NVIDIA GPU usage (requires PyTorch with CUDA installed)
  - `cpu`: Explicit CPU usage
- Whisper models: tiny, base, small, medium, large-v2, large-v3 (default)
- Compute type auto-adjusted: `int8` for CPU, `float16` for CUDA
- CUDA detection uses `torch.cuda.is_available()` when PyTorch is installed

## Testing Strategy

- Fast unit tests for core utilities (export, OCR, FFmpeg wrappers)
- Integration tests marked with `@pytest.mark.manual` (require manual validation)
- Slow tests marked with `@pytest.mark.slow`
- Test fixtures use small sample videos to minimize runtime

## Release Process

Uses CalVer versioning (YYYY.MM.DD):

```bash
# Update version in pyproject.toml
vim pyproject.toml

# Commit and tag
git add pyproject.toml
git commit -m "Bump version to 2025.10.24"
git push origin main
git tag v2025.10.24
git push origin v2025.10.24
```

GitHub Actions automatically builds and publishes to PyPI on tag push.

## Important Constants (constants.py)

- `DEFAULT_SCENE_THRESHOLD = 0.025`: FFmpeg scene filter threshold (0-1 scale)
- `DEFAULT_MIN_SCENE_LEN = 2.0`: Minimum segment duration (seconds)
- `DEFAULT_START_OFFSET = 3.0`: Skip first N seconds to avoid setup noise
- `DEFAULT_SEGMENTS_PER_HOUR = 30`: Opencast optimizer target
- `DEFAULT_WHISPER_MODEL = "large-v3"`: Best accuracy model
- `DEFAULT_DEVICE = "auto"`: Auto-detect MLX or CPU

## Code Style Requirements

- Line length: 100 characters (configured in pyproject.toml)
- Type hints required for all function signatures (`disallow_untyped_defs = true`)
- Docstrings follow Google style (Args, Returns, Raises sections)
- Ruff linter rules: E, F, I, N, W, UP (ignores E501 for line length)

## Dependencies

**Core:**
- faster-whisper: Whisper inference engine (CTranslate2 backend)
- opencv-python: Video frame extraction
- pytesseract: OCR
- yt-dlp: Video download from URLs
- tqdm: Progress bars

**Optional:**
- mlx-whisper: Apple Silicon optimized Whisper
- mlx-vlm: Apple Silicon optimized vision models (Qwen3-VL-8B-4bit)
- torch, transformers (from GitHub main), torchvision, accelerate, autoawq: PyTorch Qwen3-VL-8B
  - Uses Qwen/Qwen3-VL-8B-Instruct (~16GB float16)
  - Fits in 16GB GPU
  - Note: AWQ quantized models don't support CPU offload, so 30B AWQ won't work on 16GB GPUs

**Dev:**
- pytest, pytest-cov: Testing
- ruff: Linting and formatting
- mypy: Type checking
