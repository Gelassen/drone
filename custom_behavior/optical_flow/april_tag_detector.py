import cv2
import apriltag
import numpy as np

from custom_behavior.optical_flow.target_detection import TargetDetection

class AprilTagDetector:

    def __init__(
            self,
            tag_family = "tag16h5",
            tag_size_m=0.05,
            focal_length_px=600.0
        ):
        self.at_options = apriltag.DetectorOptions(families=tag_family)
        self.apriltag_detector = apriltag.Detector(self.at_options)

        self.tag_size_m=tag_size_m
        self.focal_length_px=focal_length_px

    def get_focal_length_px(self):
        return self.focal_length_px

    def find_squares(self, frame, min_area=2000, max_area=15000, aspect_tol=0.3):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        squares = []
        for cnt in contours:
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                x, y, w, h = cv2.boundingRect(approx)
                area = w * h
                ratio = w / float(h) if h else 0
                if area >= min_area and area <= max_area and abs(ratio - 1) < aspect_tol:
                    squares.append((x, y, w, h))
        return squares

    def detect_tags_in_roi(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return self.apriltag_detector.detect(gray)

    def estimate_distance_from_px(self, px_size):
        if px_size <= 0:
            return None
        return (self.focal_length_px * self.tag_size_m) / px_size
    
    def detect_best_target(self, frame) -> TargetDetection | None:
        squares = self.find_squares(frame)

        best = self.detect_apriltag(frame, squares)
        if best:
            return best

        return self.fallback_square(squares)
    
    def detect_apriltag(self, frame, squares):
        best = None

        for (x, y, w, h) in squares:
            roi = frame[y:y+h, x:x+w]
            tags = self.detect_tags_in_roi(roi)

            for t in tags:
                corners = np.array(t.corners)
                side = np.mean([
                    np.linalg.norm(corners[0] - corners[1]),
                    np.linalg.norm(corners[1] - corners[2])
                ])
                cx = corners[:, 0].mean() + x
                cy = corners[:, 1].mean() + y

                if not best or side > best.px_size:
                    best = TargetDetection(cx, cy, side, "tag")

        return best

    def fallback_square(self, squares):
        if not squares:
            return None

        x, y, w, h = max(squares, key=lambda s: s[2] * s[3])
        return TargetDetection(
            cx=x + w / 2,
            cy=y + h / 2,
            px_size=max(w, h),
            source="square"
        )

