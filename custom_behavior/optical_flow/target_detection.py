from dataclasses import dataclass

@dataclass
class TargetDetection:
    cx: float
    cy: float
    px_size: float
    source: str   # "tag" | "square"

    corners = None      # shape (4,2)
    h = None         # 3x3
    ts = None