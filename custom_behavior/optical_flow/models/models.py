from enum import Enum, auto

class TelemetryEvents(Enum):
    MANAGING_COMMAND = auto(),
    SCALED_COMMAND = auto(),
    APRIL_TAG_DETECTION = auto(),
    SIGNAL_CONFIDENCE = auto(),
    SIGNAL_METRICS = auto(),
    CHANNEL_CONFIDENCE = auto(),
    GATED_CHANNEL_CONFIDENCE = auto(),
    SCALE_DEBUG = auto(),
    RAW_COMMAND = auto(),
    RAW_COMMAND_GAIN = auto()