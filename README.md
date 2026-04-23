![slidegeist_logo](https://github.com/user-attachments/assets/97a1e482-d90c-41a0-a27c-a503043accad)

## Features

- **Scene detection** using global pixel difference (research-based method optimized for lecture videos)
- **Automatic slide extraction** with simple numbered filenames (slide_001, slide_002, ...)
- **Audio transcription** through a running OpenAI-compatible Whisper service
- **Markdown export** - single `slides.md` file (LLM-friendly) or split mode with separate files
- **OCR** with Tesseract
- **AI descriptions** through a running `llama.cpp` service

## Requirements

- **Python ≥ 3.10**
- **FFmpeg** (must be installed separately and available in PATH)
- **Whisper server** speaking the OpenAI `/v1/audio/transcriptions` API on `127.0.0.1:8427`
  (e.g. `whisper.cpp`'s `whisper-server`, faster-whisper-server, LocalAI, Vox-Box)
- **llama.cpp** running a completion API on `127.0.0.1:8081`

### Installing FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt-get install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html) or use:
```bash
winget install ffmpeg
```

## Installation

```bash
pip install slidegeist
```

### Developer Setup

```bash
git clone git@github.com:itpplasma/slidegeist.git
cd slidegeist
pip install -e ".[dev]"
```

## Quick Start

Process a lecture video to extract slides and transcript:

```bash
slidegeist lecture.mp4 --out output/
```

This creates:
```
output/
├── slides.md                        # Combined file with table of contents and all slides
└── slides/
    ├── slide_001.jpg                # Slide images (1-based numbering)
    ├── slide_002.jpg
    └── slide_003.jpg
```

For separate slide files (useful for navigation in some tools), use `--split`:
```bash
slidegeist lecture.mp4 --split
```

This creates:
```
output/
├── index.md                         # Overview with links to all slides
├── slide_001.md                     # Slide 1 with transcript and OCR
├── slide_002.md                     # Slide 2 with transcript and OCR
├── slide_003.md                     # Slide 3 with transcript and OCR
└── slides/
    ├── slide_001.jpg                # Slide images
    ├── slide_002.jpg
    └── slide_003.jpg
```

## Usage

### Full Processing

```bash
# Basic usage (uses the configured remote services)
slidegeist video.mp4

# Specify output directory
slidegeist video.mp4 --out my-output/

# Use smaller/faster model
slidegeist video.mp4 --model base

# Adjust scene detection sensitivity (0.0-1.0, default 0.025).
# Acts as the *starting point* for the Opencast optimizer.
# Lower values bias toward more segments; higher values toward fewer.
slidegeist video.mp4 --scene-threshold 0.015

# Explicit process command (same as default)
slidegeist process video.mp4
```

### Individual Operations

```bash
# Extract only slides (no transcription)
slidegeist slides video.mp4
```

## CLI Options

```
slidegeist <video> [options]
slidegeist {process,slides} <video> [options]

Options:
  --out DIR              Output directory (default: video filename)
  --split               Create separate markdown files (index.md + slide_NNN.md)
                        instead of single slides.md (default: combined file)
  --scene-threshold NUM  Initial scene detection sensitivity 0.0-1.0 (default: 0.025)
                         Used as the optimizer's starting threshold; it will
                         auto-adjust to reach a stable segment count.
  --model NAME          Whisper model: tiny, base, small, medium, large, large-v2, large-v3
                        (default: large-v3)
  --format FMT          Image format: jpg or png (default: jpg)
  -v, --verbose         Enable verbose logging
```

## Output Format

### Default: Combined slides.md (Recommended)

By default, Slidegeist creates a single `slides.md` file containing:
- Video metadata (source, duration, model used)
- Table of contents with clickable links to each slide
- All slides with images, transcripts, and OCR content

**Benefits:**
- Single file is easy to process with LLMs
- No navigation between files needed
- Smaller overall output size

Example structure:
```markdown
# Lecture Slides

**Video:** lecture.mp4
**Duration:** 45:30
**Transcription Model:** large-v3

## Table of Contents

- [Slide 1](#slide_001) • 00:00-05:15
- [Slide 2](#slide_002) • 05:15-12:30
...

---

## Slide 1

**Time:** 00:00 - 05:15

![Slide](slides/slide_001.jpg)

**Slide Content:**
Introduction to Quantum Mechanics

**Transcript:**
Today we discuss quantum mechanics and its implications...

---

## Slide 2
...
```

### Split Mode (--split flag)

With `--split`, creates separate files for each slide (useful for some viewers/tools):
- **Index**: `index.md` - Overview with links to individual slide files
- **Slide markdown**: `slide_001.md`, `slide_002.md`, ... - Per-slide files with YAML front matter
- **Slide images**: `slides/slide_001.jpg`, `slides/slide_002.jpg`, ...

Each split slide file contains:
```markdown
---
id: slide_001
index: 1
time_start: 0.0
time_end: 315.0
image: slides/slide_001.jpg
---

# Slide 1

[![Slide Image](slides/slide_001.jpg)](slides/slide_001.jpg)

## Transcript

Today we discuss quantum mechanics...

## Slide Content

Introduction to Quantum Mechanics

**Visual Elements:** diagram, formula
```

## How It Works

1. **Scene Detection**: Uses FFmpeg's scene filter (SAD-based) with an Opencast-style optimizer to identify slide changes
   - Iteratively adjusts the scene threshold to target ~30 segments per hour (typical slide pace)
   - Treats `--scene-threshold` as the *initial* threshold; the optimizer raises or lowers it until the slide count converges
   - Merges segments shorter than 2 seconds to suppress rapid flickers
   - Based on Opencast's VideoSegmenterService implementation
2. **Slide Extraction**: Extracts frames at 80% through each segment into `slides/` directory with simple `slide_XXX.jpg` names
3. **Transcription**: Extracts audio with FFmpeg and submits it, in 2-minute chunks, to the running OpenAI-compatible Whisper HTTP API
4. **OCR**: Uses Tesseract OCR on extracted slide images
5. **AI descriptions**: Sends OCR and transcript context to the running `llama.cpp` server
6. **Export**: Generates Markdown files with YAML front matter, linking slides to their transcripts and OCR content

## Performance

**Model Recommendations:**
- `large-v3-turbo`: Fast remote transcription when your Whisper server exposes it
- `large-v3`: Best accuracy (default) - recommended for production
- `medium`: Good balance - 2x faster, slightly lower accuracy
- `base`: Quick testing - 5x faster, noticeably lower accuracy
- `tiny`: Very fast - 10x faster, lowest accuracy

## Troubleshooting

### Remote Services

```bash
# Verify llama.cpp
curl http://127.0.0.1:8081/health

# Verify the Whisper server
curl -I http://127.0.0.1:8427/v1/audio/transcriptions
```

Set `SLIDEGEIST_LLAMACPP_URL` or `SLIDEGEIST_WHISPER_URL` if the services listen on different addresses.

## Limitations

- Scene detection may need threshold tuning for some videos (default 0.025 works well for most lectures; because the optimizer auto-adjusts, use lower values like 0.015 to bias toward more slides or 0.03+ to bias toward fewer major transitions)

### Advanced Threshold Tuning

- The Opencast optimizer targets roughly 30 segments per hour. That goal works well for standard lectures but you can steer it:
  - Lower `--scene-threshold` to encourage more segments before optimization. Useful when the optimizer consistently undershoots the actual slide count.
  - Raise `--scene-threshold` to bias toward fewer segments when the optimizer overshoots and splits slides too often.
- `--scene-threshold` is still bounded between 0.0 and 1.0. Values outside this range will be rejected by the CLI validator.
- No speaker diarization
- No automatic slide deduplication

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linter
ruff check slidegeist/

# Run type checker
mypy slidegeist/
```

## Legal Notice

Slidegeist is provided for educational and research purposes only.
Users must ensure they have the legal right to access, download, or process any video files they use with this tool.
The author does not endorse or facilitate copyright infringement or violation of platform terms of service.

## License

MIT License - Copyright (c) 2025 Christopher Albert

See [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
