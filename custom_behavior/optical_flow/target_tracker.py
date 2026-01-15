from custom_behavior.optical_flow.april_tag_detector import AprilTagDetector

class TargetTracker:
    def __init__(self, detector: AprilTagDetector, lost_threshold: int):
        self.detector = detector
        self.lost_threshold = lost_threshold
        self.lost_frames = 0
        self.ever_acquired = False

    def process(self, frame):
        detection = self.detector.detect_best_target(frame)
        if detection:
            self.lost_frames = 0
            self.ever_acquired = True
        else:
            self.lost_frames += 1
        return detection

    def is_lost(self) -> bool:
        return self.ever_acquired and self.lost_frames > self.lost_threshold
