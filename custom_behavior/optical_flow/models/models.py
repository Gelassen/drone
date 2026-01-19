from enum import Enum, auto

class TelemetryEvents(Enum):
    MANAGING_COMMAND = auto(),
    SCALED_COMMAND = auto(),
    APRIL_TAG_DETECTION = auto()