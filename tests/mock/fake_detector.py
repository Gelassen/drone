
class FakeDetector:
    def __init__(self, detections):
        self.detections = detections
        self.idx = 0

    def detect_best_target(self, frame):
        if self.idx < len(self.detections):
            d = self.detections[self.idx]
            self.idx += 1
            return d
        return None

    def estimate_distance_from_px(self, px):
        return 1.5

    def get_focal_length_px(self):
        return 500
