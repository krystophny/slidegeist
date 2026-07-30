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

# Output formats
DEFAULT_IMAGE_FORMAT = "jpg"
DEFAULT_OUTPUT_DIR = "output"
