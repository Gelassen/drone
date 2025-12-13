from dataclasses import dataclass

@dataclass
class TargetDetection:
    cx: float
    cy: float
    px_size: float
    source: str   # "tag" | "square"