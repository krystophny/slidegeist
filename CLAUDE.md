# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Slidegeist extracts slides and timestamped transcripts from lecture videos using FFmpeg scene detection and external services for transcription (Voxtype STT) and AI slide descriptions (llama.cpp with Qwen3.5). The project emphasizes minimal dependencies and service-based architecture.

## Build, Test, and Lint Commands

### Development Setup
```bash
uv pip install -e ".[dev]"       # Install in editable mode with dev dependencies
```

### Services Setup
```bash
scripts/setup-voxtype-stt.sh     # Start Voxtype STT service (port 8427)
scripts/setup-local-llm.sh       # Start llama.cpp with Qwen3.5-9B (port 8081)
```

### Testing
```bash
uv run pytest                    # Run all tests (excludes manual tests)
uv run pytest -v                 # Verbose output
uv run pytest tests/test_export.py  # Run specific test file
uv run pytest -m "not slow"     # Skip slow tests
```

### Code Quality
```bash
uv run ruff check slidegeist/    # Run linter
uv run ruff check --fix slidegeist/  # Auto-fix linting issues
uv run ruff format slidegeist/   # Auto-format code
```

### Running the CLI
```bash
slidegeist video.mp4                        # Process video (default: slides + transcript)
slidegeist video.mp4 --out output/          # Specify output directory
slidegeist video.mp4 --scene-threshold 0.02 # Adjust scene detection sensitivity
slidegeist video.mp4 --model large-v3-turbo # Whisper model name
slidegeist video.mp4 --stt-url http://host:8427  # Custom STT service URL
slidegeist video.mp4 --llm-url http://host:8081  # Custom LLM service URL
slidegeist slides video.mp4                 # Extract only slides (no transcription)
```

## Architecture

### Service-Based Design

Slidegeist uses external services for heavy ML workloads instead of bundling models:

- **Voxtype STT** (port 8427): OpenAI-compatible speech-to-text API using whisper-rs
  - Endpoint: `POST /v1/audio/transcriptions`
  - Default model: large-v3-turbo
  - Requires branch `feature/single-daemon-openai-stt-api` of voxtype

- **llama.cpp** (port 8081): OpenAI-compatible LLM/VLM API
  - Endpoint: `POST /v1/chat/completions`
  - Default model: Qwen3.5-9B (Q4_K_M quantization)
  - Supports vision via base64 image URLs

### Processing Pipeline (pipeline.py)

The main `process_video()` function orchestrates processing with smart resume capabilities:

**Smart Resume**: If output directory contains both a video file and a slides subdirectory with images, automatically skips slide extraction and resumes from transcription.

**Processing Steps**:

1. **Scene Detection** (ffmpeg_scene.py): Uses FFmpeg's SAD-based scene filter with Opencast-style optimization
   - Iteratively adjusts threshold to target ~30 segments/hour (typical lecture pace)
   - Merges segments shorter than 2 seconds to filter out rapid flickers
   - `--scene-threshold` serves as the optimizer's starting point

2. **Slide Extraction** (slides.py): Extracts frames at 80% through each detected segment
   - Simple numbered filenames: slide_001.jpg, slide_002.jpg, etc.
   - Supports JPG and PNG formats

3. **Transcription** (transcribe.py): Sends audio to Voxtype STT service
   - Extracts audio via FFmpeg, sends WAV to OpenAI-compatible API
   - Returns word-level timestamps and segments

4. **OCR** (ocr.py): Optional Tesseract OCR for text extraction
   - Uses pytesseract with English+German language support

5. **AI Descriptions** (ai_description.py): Vision-language slide analysis via llama.cpp
   - Sends slide images as base64 to the LLM server
   - Generates structured 5-section descriptions for reconstruction

6. **Export** (export.py): Generates Markdown files with YAML front matter
   - Default: Single `slides.md` with table of contents (LLM-friendly)
   - Split mode (`--split`): Separate files per slide with `index.md`

### Key Design Decisions

- **Opencast compatibility**: Scene detection threshold and optimization mirror Opencast's VideoSegmenterService implementation
- **Service architecture**: No bundled ML models, uses external llama.cpp and voxtype services
- **Minimal dependencies**: Core package needs only FFmpeg, opencv-python, pytesseract, Pillow
- **Research-based defaults**: Scene threshold (0.025), target segments/hour (30), minimum segment length (2s) are based on Opencast research

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
- `DEFAULT_WHISPER_MODEL = "large-v3-turbo"`: Default Whisper model
- `DEFAULT_STT_URL = "http://127.0.0.1:8427"`: Voxtype STT service
- `DEFAULT_LLM_URL = "http://127.0.0.1:8081"`: llama.cpp server

## Code Style Requirements

- Line length: 100 characters (configured in pyproject.toml)
- Type hints required for all function signatures (`disallow_untyped_defs = true`)
- Docstrings follow Google style (Args, Returns, Raises sections)
- Ruff linter rules: E, F, I, N, W, UP (ignores E501 for line length)

## Dependencies

**Core (lightweight):**
- numpy: Array operations
- opencv-python: Video frame extraction
- pytesseract: OCR (optional, degrades gracefully)
- yt-dlp: Video download from URLs
- Pillow: Image processing
- tqdm: Progress bars
- psutil: System resource detection

**External Services (not Python dependencies):**
- Voxtype STT: whisper-rs based transcription daemon
- llama.cpp: LLM/VLM server with Qwen3.5-9B

**Dev:**
- pytest, pytest-cov: Testing
- ruff: Linting and formatting
- mypy: Type checking
