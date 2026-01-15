from dataclasses import dataclass

@dataclass
class TargetDetection:
    cx: float
    cy: float
    px_size: float
    source: str   # "tag" | "square"
    side = None
    corners = None      # shape (4,2)
    homography = None         # 3x3
    timestamp = None