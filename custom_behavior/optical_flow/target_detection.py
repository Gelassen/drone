import numpy as np

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class TargetDetection:
    cx: float
    cy: float
    px_size: float | None = None
    source: str | None = None  # "tag" | "square"
    side: Optional[float] = None
    corners: Optional[np.ndarray] = None   # (4,2)
    homography: Optional[np.ndarray] = None
    timestamp: Optional[float] = None