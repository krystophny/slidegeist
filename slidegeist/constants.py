"""Constants used across the slidegeist package."""

# Scene detection - Opencast FFmpeg-based (default implementation)
# Based on: https://docs.opencast.org/r/4.x/admin/modules/videosegmentation/
DEFAULT_SCENE_THRESHOLD = 0.025  # FFmpeg scene filter threshold (0-1 scale, SAD-based)
                                 # Opencast default: 0.025 (2.5% pixel change)
                                 # Lower = more sensitive. Typical range: 0.01-0.05
DEFAULT_MIN_SCENE_LEN = 2.0  # Minimum segment length in seconds (stability threshold)
                             # Opencast default: 60s, adapted to 2s for slide detection
DEFAULT_START_OFFSET = 3.0  # Skip first N seconds to avoid mouse movement during setup

# Opencast optimization parameters (target slides/hour mirrors Opencast default)
# Research shows typical presentations have 15-45 slides/hour, 30 is a good middle ground
DEFAULT_SEGMENTS_PER_HOUR = 30  # Target segments per hour (matches typical presentation pace)
DEFAULT_MAX_ERROR = 0.25     # Maximum error tolerance (25%, Opencast default)
DEFAULT_MAX_CYCLES = 3       # Maximum optimization iterations (Opencast default)

# Whisper transcription
DEFAULT_WHISPER_MODEL = "large-v3-turbo"  # Fast and accurate (voxtype default)
DEFAULT_DEVICE = "auto"  # Auto-detect: service first, then MLX, then CPU

# Service URLs (OpenAI-compatible endpoints)
DEFAULT_STT_URL = "http://127.0.0.1:8427"   # Voxtype STT service
DEFAULT_LLM_URL = "http://127.0.0.1:8081"   # llama.cpp server (Qwen3.5-9B)

# Output formats
DEFAULT_IMAGE_FORMAT = "jpg"
DEFAULT_OUTPUT_DIR = "output"
