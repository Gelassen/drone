import cv2
import apriltag

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