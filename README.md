![slidegeist_logo](https://github.com/user-attachments/assets/97a1e482-d90c-41a0-a27c-a503043accad)

## Features

- **Transition detection** using an adaptive visual-evidence ensemble without slide-cadence assumptions
- **Automatic slide extraction** with simple numbered filenames (slide_001, slide_002, ...)
- **Instructional-state filtering** that excludes desktop/navigation/recording UI with auditable hashes
- **Audio transcription** through a running OpenAI-compatible Whisper service
- **Markdown export** - single `slides.md` file (LLM-friendly) or split mode with separate files
- **OCR** with Tesseract
- **Multimodal AI descriptions** through a running OpenAI-compatible `llama.cpp` service

## Requirements

- **Python ≥ 3.10**
- **FFmpeg** (must be installed separately and available in PATH)
- **Whisper server** speaking the OpenAI `/v1/audio/transcriptions` API on `127.0.0.1:8427`
  (e.g. `whisper.cpp`'s `whisper-server`, faster-whisper-server, LocalAI, Vox-Box)
- **llama.cpp** running a multimodal chat-completion API on `127.0.0.1:8081`

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

# Adjust transition sensitivity (0.0-1.0, default 0.025).
# This only biases evidence sensitivity; it never targets a slide count.
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
  --scene-threshold NUM  Transition sensitivity bias 0.0-1.0 (default: 0.025).
                         Lower is more sensitive; slide count is never targeted.
  --model NAME          Whisper model: tiny, base, small, medium, large, large-v2,
                        large-v3, large-v3-turbo
                        (default: large-v3-turbo)
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

1. **Transition detection**: Fuses structural similarity, perceptual hash, HSV histogram, edge change, and spatial coverage
   - Establishes robust median/MAD baselines for each video
   - Requires complementary evidence to reject presenter motion and brightness flashes
   - Uses timing only to merge duplicate peaks from the same transition
   - Writes `transition_detection.json` with auditable scores and thresholds
2. **Slide Extraction**: Extracts frames at 80% through each segment into `slides/` directory with simple `slide_XXX.jpg` names
3. **Transcription**: Extracts audio with FFmpeg and submits it, in 2-minute chunks, to the running OpenAI-compatible Whisper HTTP API
4. **OCR**: Uses Tesseract OCR on extracted slide images
5. **AI descriptions**: Sends the slide image plus OCR and transcript context to the configured multimodal server
6. **Export**: Generates Markdown files with YAML front matter, linking slides to their transcripts and OCR content

Combined and split exports both checkpoint AI descriptions atomically and
resume without regenerating valid classified frames. A required stage that
remains failed exits nonzero; use `--retry-failed` after restoring its service.

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
Set `SLIDEGEIST_LLAMACPP_MODEL` when the endpoint exposes more than one model.
`SLIDEGEIST_LLAMACPP_MAX_IMAGE_DIMENSION` controls the longest edge of the
multimodal request image (default: 1024). This reduces service latency without
changing the full-resolution extracted slide stored on disk.
`SLIDEGEIST_LLAMACPP_MAX_TOKENS` controls the reconstruction response ceiling
(default: 1024; minimum: 256). Formula-heavy slides may need the default.

For example:

```bash
export SLIDEGEIST_LLAMACPP_URL=http://model-host:8080
export SLIDEGEIST_LLAMACPP_MODEL=qwen27b
export SLIDEGEIST_LLAMACPP_MAX_IMAGE_DIMENSION=1024
export SLIDEGEIST_LLAMACPP_MAX_TOKENS=1024
export SLIDEGEIST_WHISPER_URL=http://127.0.0.1:8427
```

## Transcription providers

Two backends. **The local Whisper server is the default**; Voxtral (Mistral) is
opt-in with `--transcriber voxtral`.

```bash
slidegeist lecture.mp4                    # local Whisper + local descriptions
slidegeist lecture.mp4 --cloud            # Voxtral audio + Gemma 4 descriptions
slidegeist lecture.mp4 --diarize local    # force DiariZen for speaker labels
slidegeist lecture.mp4 --diarize off      # no speaker labels
```

There is deliberately **no automatic fallback** between providers: a missing
`MISTRAL_API_KEY` is an error telling you to drop `--cloud`, never a silent switch that would
upload audio meant to stay local (or bill for a run expected to be free). Both
preconditions are checked before any download starts.

| Variable | Default | Purpose |
|---|---|---|
| `MISTRAL_API_KEY` | – | Voxtral auth (vendor-standard name) |
| `SLIDEGEIST_MISTRAL_API_KEY` | – | Override, checked first |
| `SLIDEGEIST_MISTRAL_URL` | `https://api.mistral.ai` | Proxy/gateway override |
| `SLIDEGEIST_VOXTRAL_MODEL` | `voxtral-mini-latest` | Model id |
| `SLIDEGEIST_VOXTRAL_MAX_CHUNK_S` | `3000` | Split threshold for long audio |
| `SLIDEGEIST_TRANSCRIBER` | `whisper` | Default backend |
| `SLIDEGEIST_DIARIZEN_PYTHON` | – | Interpreter with DiariZen installed |
| `SLIDEGEIST_DIARIZEN_MODEL` | `BUT-FIT/diarizen-wavlm-large-s80-md` | Diarization model |
| `SLIDEGEIST_DIARIZEN_DEVICE` | `auto` | `cpu`, `cuda` or `auto` |
| `SLIDEGEIST_DIARIZEN_TIMEOUT` | `7200` | Subprocess timeout (seconds) |

### Provider trade-off

Voxtral **cannot return word timing and speaker labels together**: setting
`diarize=true` forces segment granularity. Speaker labels win. When word-level
timing matters, use `--local`, which gets word timing from whisper.cpp and
speaker turns from DiariZen independently.

## Slide description providers

Slide descriptions default to the **local llama.cpp server**. Gemma 4 via
OpenRouter is opt-in with `--describer openrouter`.

```bash
slidegeist lecture.mp4                          # fully local (default)
slidegeist lecture.mp4 --cloud                  # Voxtral + Gemma 4 (opt-in)
```

⚠️ **Do not make Gemma the pipeline default.** Over a full 40-frame lecture it
classified **36-38 of 40 genuine teaching pages as NON-SLIDE**, because the
frames are Goodnotes screenshots carrying a macOS menu bar and a screen-sharing
banner. The frame filter and the description share one call, so that verdict
*deletes* the frame. Qwen3.6-27B classified all 40 correctly. Gemma is only safe
behind a separate classification pass.

| Variable | Default | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | – | Describer auth (vendor-standard name) |
| `SLIDEGEIST_OPENROUTER_API_KEY` | – | Override, checked first |
| `SLIDEGEIST_OPENROUTER_URL` | `https://openrouter.ai/api` | Gateway override |
| `SLIDEGEIST_OPENROUTER_MODEL` | `google/gemma-4-26b-a4b-it` | Vision model id |
| `SLIDEGEIST_DESCRIBER` | `local` | Default backend |

As with transcription, there is **no automatic fallback**: a missing key is an
error telling you to drop `--cloud`.

### What the measurements showed

Measured on real handwritten lecture pages rather than chosen from a price table:

- It encodes a 1024 px slide in **~320 tokens**; Qwen 3.6 needs **~5,770** for the
  same image, and its reasoning modes consumed 2,500 output tokens without
  producing a transcription at all.
- It reads handwritten physics notation correctly. Cheaper tiers (Ministral 3B,
  Mistral Small, Qwen3-VL-8B, Gemini 2.5 Flash-Lite, GPT-5-mini) misread `r₉₀`
  as `Γ₉₀` — a silent, physics-substantive error.
- In an 8-slide A/B judged against the source images, `gemma-4-26b-a4b`
  beat the dense `gemma-4-31b` 6-2, at roughly half the latency. Most wins were
  for *not inventing* content that was absent from the page.

But over a whole lecture it rejected almost every frame as non-instructional,
which is why it is not the default. Transcription accuracy and frame
classification are separate abilities, and only the second one is load-bearing
for a pipeline that deletes what it rejects.

On audio, Voxtral corrupted the lecture's central proper noun - "Boltzmann"
became "Ortsmann" or "Bortzman" in 5 of 10 occurrences - where local whisper got
all 10 right. Voxtral remains better on ordinary speech and is the only option
with built-in diarization, so it stays available; it is not the default.

### Speaker diarization

Local diarization uses [DiariZen](https://github.com/BUTSpeechFIT/DiariZen) in a
**separate interpreter**, for two reasons: torch has no wheels for Python 3.14,
and DiariZen's pretrained weights are CC BY-NC 4.0 (non-commercial), so they
must not become a dependency of this MIT-licensed package. Slidegeist only ever
shells out to an interpreter you configure.

```bash
python3.11 -m venv ~/.venvs/diarizen311
git clone --recurse-submodules https://github.com/BUTSpeechFIT/DiariZen
# follow DiariZen's install instructions inside that venv, then:
export SLIDEGEIST_DIARIZEN_PYTHON=~/.venvs/diarizen311/bin/python
```

Diarization answers *when each voice speaks*, not *who they are*: labels are
`SPEAKER_00`, `SPEAKER_01`, … and are local to one recording.

## Limitations

- Transition detection can still miss very small incremental builds or mistake a full-screen animation for a slide. Inspect `transition_detection.json` and extracted frames.

See [the transition-detection design and oracle benchmark](docs/transition-detection.md).
 - `--scene-threshold` is still bounded between 0.0 and 1.0. Values outside this range will be rejected by the CLI validator.
 - Speaker diarization identifies *how many* voices and *when* they speak, not
   *who* they are. Labels are local to one recording; recognising the same
   person across recordings would need a separate enrollment step.
 - Voxtral cannot return word timing together with speaker labels; use
   `--local` when word-level timing matters.
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
