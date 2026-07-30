# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Slidegeist extracts slides and timestamped transcripts from lecture videos using FFmpeg scene detection and two locally hosted HTTP services: a llama.cpp server for text generation and an OpenAI-compatible Whisper server (e.g. whisper.cpp's `whisper-server`) for transcription. No ML dependencies are installed in-process; slidegeist is effectively a small orchestrator around FFmpeg, Tesseract, and those two REST APIs.

## Build, Test, and Lint Commands

### Development Setup
```bash
pip install -e ".[dev]"          # Install in editable mode with dev dependencies
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
```

### Running the CLI
```bash
slidegeist video.mp4                        # Process video (default: slides + transcript)
slidegeist video.mp4 --out output/          # Specify output directory
slidegeist video.mp4 --scene-threshold 0.02 # Adjust scene detection sensitivity
slidegeist slides video.mp4                 # Extract only slides (no transcription)
```

## External Services

Slidegeist talks to two locally hosted OpenAI-compatible HTTP endpoints. Both must be reachable before the pipeline is run; there is no embedded fallback.

| Service         | Default URL              | Env override                  | Purpose                                  |
|-----------------|--------------------------|-------------------------------|------------------------------------------|
| llama.cpp       | `http://127.0.0.1:8081`  | `SLIDEGEIST_LLAMACPP_URL`     | Multimodal slide descriptions            |
| Whisper server  | `http://127.0.0.1:8427`  | `SLIDEGEIST_WHISPER_URL`      | OpenAI-compatible audio transcription    |

All HTTP is implemented in `slidegeist/services.py` using the stdlib only.
Health probes hit `/health` first; Whisper falls back to probing
`/v1/audio/transcriptions` for compatible servers without a health route.
For multimodal requests, `SLIDEGEIST_LLAMACPP_MAX_IMAGE_DIMENSION` limits the
longest edge of the service copy (default 1024 pixels) while preserving the
full-resolution extracted frame on disk.

### Recommended Whisper server: whisper.cpp

The canonical local setup on Arch/CachyOS:

```bash
pacman -S whisper.cpp-cuda                               # or whisper.cpp for CPU-only
yay -S whisper.cpp-model-large-v3-turbo-q5_0

whisper-server \
  --model /usr/share/whisper.cpp-model-large-v3-turbo-q5_0/ggml-large-v3-turbo-q5_0.bin \
  --host 127.0.0.1 --port 8427 \
  --inference-path /v1/audio/transcriptions \
  --convert --threads 4
```

A ready-made user unit lives at `~/.config/systemd/user/whisper-server.service` on the developer machine. `whisper.cpp`'s server returns full `verbose_json` with per-segment and per-word timestamps, which slidegeist needs for slide/transcript alignment.

### Drop-in alternatives

Any server that implements `POST /v1/audio/transcriptions` with `response_format=verbose_json` and populated `segments`/`words` will work. Point slidegeist at it with `SLIDEGEIST_WHISPER_URL`:

- **[faster-whisper-server](https://github.com/fedirz/faster-whisper-server)** — CTranslate2 backend, OpenAI-compatible, Docker image available.
- **[LocalAI](https://localai.io/features/audio-to-text/)** — generic OpenAI-compatible gateway that can serve Whisper (plus llama.cpp and more) from a single daemon.
- **[Vox-Box](https://github.com/gpustack/vox-box)** — OpenAI-compatible audio server with Whisper turbo support.

Before switching servers, verify the response shape:

```bash
curl -s -F "model=large-v3-turbo" -F "response_format=verbose_json" \
     -F "timestamp_granularities[]=segment" -F "timestamp_granularities[]=word" \
     -F "file=@clip.wav" http://127.0.0.1:8427/v1/audio/transcriptions | jq 'keys'
```

The response must contain a non-empty `segments` array with real `start`/`end` values. Without it, slide-level transcript windows collapse to zero width.

## Architecture

### Processing Pipeline (pipeline.py)

The main `process_video()` function orchestrates processing with smart resume capabilities:

**Smart Resume**: If the output directory contains both a video file and a slides subdirectory with images, slide extraction is skipped and processing resumes from transcription. This lets you re-run slidegeist against the same directory to add transcription to existing slides. Combined and split Markdown exports checkpoint descriptions atomically. A required stage that remains failed exits nonzero; `--retry-failed` retries it after service recovery.

**Processing Steps**:

1. **Transition Detection** (`transition_detector.py`): Fuses SSIM-like structure, perceptual hash, HSV histogram, edges, and spatial coverage.
   - Robust per-video baselines replace cadence assumptions.
   - `--scene-threshold` is a bounded sensitivity bias, never a count target.
   - A terminal flash or recording fade is rejected when no later sample supports a stable scene.
   - `transition_detection.json` records evidence and thresholds.

2. **Slide Extraction** (`slides.py`): Extracts frames at 80% through each detected segment.
   - Simple numbered filenames: `slide_001.jpg`, `slide_002.jpg`, ...
   - Supports JPG and PNG formats.

3. **Transcription** (`transcribe.py` + `services.whisper_transcribe`): Extracts mono 16 kHz WAV, splits into 120 s chunks to stay within server upload limits, POSTs each chunk to `/v1/audio/transcriptions` with `verbose_json`, and reassembles segments/words with offset-corrected timestamps.

4. **OCR** (`ocr.py`): Tesseract only (`eng+deu`, PSM 1). The OCR result is supporting context for the multimodal describer in (5).

5. **AI Slide Description** (`ai_description.py` + `services.llama_cpp_complete`): Sends the slide image with OCR text and the transcript window to a multimodal llama.cpp server. `SLIDEGEIST_LLAMACPP_MODEL` can select a model explicitly; otherwise the first `/v1/models` entry is used. Oversized images are reduced only in the HTTP request to avoid spending vision-token compute on pixels beyond normal slide-reading resolution.

   The same response classifies the visual state. Desktop navigation, file
   browsers, dialogs, recording controls, loading screens, and teaching pages
   obscured by operating-system UI are excluded from the deck. Raw boundaries,
   rejection reasons, and frame hashes remain in `transition_detection.json`.
   Missing or contradictory classifications remain resumable, and interrupted
   rejection renames are recovered from the durable diagnostic plan.

6. **Export** (`export.py`): Generates Markdown with YAML front matter.
   - Default: single `slides.md` with table of contents (LLM-friendly).
   - Split mode (`--split`): separate files per slide with `index.md`.

### Key Design Decisions

- **Cadence-free detection**: visual evidence decides the count; expected slides/time is never a hard limit or stopping condition.
- **Out-of-process inference**: All model compute lives in external services. The package has no torch/transformers/faster-whisper/MLX dependency.
- **Research-based evidence**: see `docs/transition-detection.md` and the behavioral oracle benchmark.
- **Minimal dependencies**: Core runtime is FFmpeg, Tesseract, numpy, opencv-python, yt-dlp, Pillow, tqdm, psutil.

### Scene Detection Implementation

Two complementary implementations exist:

1. **transition_detector.py** (default): robust multimodal visual evidence and local peak selection.

2. **ffmpeg_scene.py** and **pixel_diff_detector.py** (legacy/experimental): retained for comparison and threshold-sweep research.

## Testing Strategy

- Fast unit tests for core utilities (export, OCR, FFmpeg wrappers).
- Integration tests marked with `@pytest.mark.manual` (require manual validation).
- Slow tests marked with `@pytest.mark.slow`.
- Test fixtures use small sample videos to minimize runtime.

## Release Process

Uses CalVer versioning (YYYY.MM.DD):

```bash
vim pyproject.toml
git add pyproject.toml
git commit -m "Bump version to 2026.04.23"
git push origin main
git tag v2026.04.23
git push origin v2026.04.23
```

GitHub Actions automatically builds and publishes to PyPI on tag push.

## Important Constants (constants.py)

- `DEFAULT_SCENE_THRESHOLD = 0.025`: FFmpeg scene filter threshold (0–1 scale).
- `DEFAULT_MIN_SCENE_LEN = 0.75`: Transition-burst refractory period (seconds), not a slide-duration prior.
- `DEFAULT_START_OFFSET = 3.0`: Skip first N seconds to avoid setup noise.
- `DEFAULT_SEGMENTS_PER_HOUR = 30`: legacy experimental Opencast detector only.
- `DEFAULT_WHISPER_MODEL = "large-v3-turbo"`: Model name passed to the Whisper server.
- `DEFAULT_LLAMACPP_URL = "http://127.0.0.1:8081"`
- `DEFAULT_WHISPER_URL = "http://127.0.0.1:8427"`

## Code Style Requirements

- Line length: 100 characters (configured in `pyproject.toml`).
- Type hints required for all function signatures (`disallow_untyped_defs = true`).
- Docstrings follow Google style (Args, Returns, Raises sections).
- Ruff rules: E, F, I, N, W, UP (ignores E501).

## Dependencies

**Runtime:**
- numpy, opencv-python: frame handling.
- yt-dlp: video download from URLs.
- pytesseract: OCR (wraps the system Tesseract binary).
- Pillow, tqdm, psutil: imaging, progress, process utilities.
- FFmpeg (system binary): scene detection and audio extraction.

**External services (not Python deps, must be running):**
- A llama.cpp server with any chat/completion model — Qwen-family recommended.
- A Whisper-compatible STT server exposing `/v1/audio/transcriptions` (whisper.cpp's `whisper-server` recommended).

**Dev:**
- pytest, pytest-cov, ruff, mypy.
