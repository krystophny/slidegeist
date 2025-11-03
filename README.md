![slidegeist_logo](https://github.com/user-attachments/assets/97a1e482-d90c-41a0-a27c-a503043accad)

## Features

- **Scene detection** using global pixel difference (research-based method optimized for lecture videos)
- **Automatic slide extraction** with simple numbered filenames (slide_001, slide_002, ...)
- **Audio transcription** with Whisper large-v3 model (highest quality)
- **MLX acceleration** on Apple Silicon Macs for 2-3x faster transcription
- **Markdown export** - single `slides.md` file (LLM-friendly) or split mode with separate files
- **OCR with refinement** - Tesseract OCR with optional Qwen3-VL vision model enhancement

## Requirements

- **Python ≥ 3.10**
- **FFmpeg** (must be installed separately and available in PATH)

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

**Optional AI features:**
- Apple Silicon: `pip install "slidegeist[mlx]"` for MLX acceleration and Qwen3-VL descriptions
- CUDA/CPU: `pip install "slidegeist[torch]"` for PyTorch-based Qwen3-VL descriptions
- Enable Whisper CUDA builds by installing PyTorch with CUDA before Slidegeist

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
# Basic usage (auto-detects MLX on Apple Silicon, uses large-v3 model)
slidegeist video.mp4

# Specify output directory
slidegeist video.mp4 --out my-output/

# Use GPU explicitly (NVIDIA)
slidegeist video.mp4 --device cuda

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
  --device NAME         Device: cpu, cuda, or auto (default: auto)
                        auto = MLX on Apple Silicon if available, else CPU
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
3. **Transcription**: Uses Whisper large-v3 for state-of-the-art speech-to-text with timestamps
   - Auto-detects and uses MLX on Apple Silicon for 2-3x speedup
   - Falls back to faster-whisper on other platforms
4. **OCR** (optional): Uses Tesseract OCR with optional Qwen3-VL refinement (MLX only on Apple Silicon)
5. **Export**: Generates Markdown files with YAML front matter, linking slides to their transcripts and OCR content

## Performance

**Transcription Speed (Apple Silicon with MLX):**
- 1 hour lecture: ~10-15 minutes (large-v3 model)
- Without MLX: ~25-35 minutes

**Model Recommendations:**
- `large-v3`: Best accuracy (default) - recommended for production
- `medium`: Good balance - 2x faster, slightly lower accuracy
- `base`: Quick testing - 5x faster, noticeably lower accuracy
- `tiny`: Very fast - 10x faster, lowest accuracy

## Troubleshooting

### macOS Hard Crashes with MLX

If Python crashes hard (macOS crash dialog) when using MLX on Apple Silicon:

```bash
# Disable MLX and use CPU-based transcription instead
export SLIDEGEIST_DISABLE_MLX=1
slidegeist video.mp4
```

This forces CPU mode even on Apple Silicon. The crash typically happens when:
- MLX libraries are corrupted or partially installed
- Metal framework has compatibility issues
- System libraries are out of date

To fix permanently, try:
```bash
# Reinstall MLX packages
pip uninstall mlx-whisper mlx-vlm -y
pip install --no-cache-dir mlx-whisper mlx-vlm
```

You can also explicitly use CPU mode without the environment variable:
```bash
slidegeist video.mp4 --device cpu
```

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
