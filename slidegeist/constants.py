"""Constants used across the slidegeist package."""

# Scene detection - robust visual ensemble. 0.025 is a neutral compatibility
# value; lower/higher values only bias evidence sensitivity and never set a
# target slide count.
DEFAULT_SCENE_THRESHOLD = 0.025
DEFAULT_MIN_SCENE_LEN = 0.75  # Collapse a transition burst, not short legitimate slides
DEFAULT_START_OFFSET = 3.0  # Skip first N seconds to avoid mouse movement during setup

# Legacy experimental Opencast detector parameters. The default detector does
# not use any of these cadence priors.
DEFAULT_SEGMENTS_PER_HOUR = 30  # Target segments per hour (matches typical presentation pace)
DEFAULT_MAX_ERROR = 0.25     # Maximum error tolerance (25%, Opencast default)
DEFAULT_MAX_CYCLES = 3       # Maximum optimization iterations (Opencast default)

# Whisper transcription
DEFAULT_WHISPER_MODEL = "large-v3-turbo"
DEFAULT_LLAMACPP_URL = "http://127.0.0.1:8081"
DEFAULT_WHISPER_URL = "http://127.0.0.1:8427"

# Remote transcription (Mistral Voxtral). Voxtral is the default provider;
# the local Whisper server stays available through --local / --transcriber whisper.
DEFAULT_MISTRAL_URL = "https://api.mistral.ai"
DEFAULT_VOXTRAL_MODEL = "voxtral-mini-latest"
# Mistral accepts up to 3 h per request. 50 min keeps a wide margin and bounds
# the cost of a retry, while still sending a typical lecture in one piece.
DEFAULT_VOXTRAL_MAX_CHUNK_S = 3000
DEFAULT_TRANSCRIBER = "whisper"

# Slide description defaults to the local llama.cpp server (Qwen 3.6-27B).
# OpenRouter/Gemma 4 remains available via --describer openrouter.
#
# Gemma transcribes formulas well - it was the only model to render a dense
# handwritten derivation page exactly - but it is NOT safe as the pipeline
# default: over a full 40-frame lecture it classified 36-38 of 40 genuine
# teaching pages as NON-SLIDE, because these are Goodnotes screenshots carrying
# a macOS menu bar and a screen-sharing banner. Since the frame filter and the
# description share one call, that verdict deletes the frame. Qwen classified
# all 40 correctly.
#
# Use Gemma only behind a separate classification pass.
DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api"
DEFAULT_DESCRIBER = "local"
DEFAULT_OPENROUTER_MODEL = "google/gemma-4-26b-a4b-it"
DEFAULT_LOCAL_DESCRIBE_MODEL = "qwen27b"
# Remote description runs several requests in flight; the local llama.cpp
# server has one slot and is pinned to 1 in get_describe_concurrency().
DEFAULT_DESCRIBE_CONCURRENCY = 8

# Speaker diarization. DiariZen runs in a separate interpreter: its weights are
# CC BY-NC 4.0 and torch has no wheels for the 3.14 host interpreter.
DEFAULT_DIARIZEN_MODEL = "BUT-FIT/diarizen-wavlm-large-s80-md"
DEFAULT_DIARIZE_MODE = "auto"
DEFAULT_DIARIZEN_TIMEOUT = 7200.0

# Output formats
DEFAULT_IMAGE_FORMAT = "jpg"
DEFAULT_OUTPUT_DIR = "output"
