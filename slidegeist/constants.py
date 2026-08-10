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
DEFAULT_VOXTRAL_MODEL = "voxtral-mini-transcribe-2602"
# Mistral accepts up to 3 h per request. 50 min keeps a wide margin and bounds
# the cost of a retry, while still sending a typical lecture in one piece.
DEFAULT_VOXTRAL_MAX_CHUNK_S = 3000
DEFAULT_TRANSCRIBER = "voxtral"

# Speaker diarization. DiariZen runs in a separate interpreter: its weights are
# CC BY-NC 4.0 and torch has no wheels for the 3.14 host interpreter.
DEFAULT_DIARIZEN_MODEL = "BUT-FIT/diarizen-wavlm-large-s80-md"
DEFAULT_DIARIZE_MODE = "auto"
DEFAULT_DIARIZEN_TIMEOUT = 7200.0

# Output formats
DEFAULT_IMAGE_FORMAT = "jpg"
DEFAULT_OUTPUT_DIR = "output"
