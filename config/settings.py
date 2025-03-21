"""
Settings and default configurations for the data splitting utility.
"""

# Split ratios
TRAIN_RATIO = 0.9
VAL_RATIO = 0.0
TEST_RATIO = 0.1

# Random seed for reproducibility
RANDOM_SEED = 42

# File operation settings
COPY_FILES = False

# File extensions filter
FILE_EXTENSIONS = None

# Default directories
DEFAULT_INPUT_DIR = (
    None  # Set to None to require input or specify a default path like "D:/Data/Input"
)
DEFAULT_OUTPUT_DIR = (
    None  # Set to None to require input or specify a default path like "D:/Data/Output"
)

# Logging settings
LOGGING_LEVELS = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}

DEFAULT_LOG_LEVEL = "INFO"
