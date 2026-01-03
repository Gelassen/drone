import numpy as np

from target_detection import TargetDetection

class Point:

    def __init__(self, x, y):
        self.x = x 
        self.y = y

class Axis:
    x_axis: float
    y_axis: float
    x_angle: float
    y_angle: float


class Aspect:
    width: float
    height: float
    aspect: float

class Speed:
    vx: float 
    vy: float
    rotation_speed: float # omega
    prev_marker: TargetDetection

class SignalsUtil:

    def __init__(self):
        print("=== SignalsUtil init ===")

    def detect_center(self, marker: TargetDetection):
        return Point(marker.cx, marker.cy) 
    
    def detect_axis(self, marker: TargetDetection):
        if marker.corners is None:
            return None

        c = marker.corners

        x_axis = c[1] - c[0]
        y_axis = c[3] - c[0]

        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        return Axis(
            x_axis, 
            y_axis, 
            np.arctan2(x_axis[1], x_axis[0]),
            np.arctan2(y_axis[1], y_axis[0])
        )
    
    def detect_aspect(self, marker: TargetDetection):
        if marker.corners is None:
            return None

        c = marker.corners

        w1 = np.linalg.norm(c[1] - c[0])
        w2 = np.linalg.norm(c[2] - c[3])
        h1 = np.linalg.norm(c[3] - c[0])
        h2 = np.linalg.norm(c[2] - c[1])

        width = (w1 + w2) / 2
        height = (h1 + h2) / 2

        aspect = width / height

        return Aspect(
            width=width,
            height=height,
            aspect=aspect
        )
    
    def detect_skew(self, marker):
        c = marker.corners

        v1 = c[1] - c[0]
        v2 = c[3] - c[0]

        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)

        skew = np.dot(v1, v2)  # 0 → идеально ортогонально

        return skew
    
    def detect_target_speed(self, prev_marker: TargetDetection, marker: TargetDetection):
        if prev_marker is None:
            prev_marker = marker
            return None

        dt = marker.ts - prev_marker.ts
        if dt <= 0:
            return None

        vx = (marker.cx - prev_marker.cx) / dt
        vy = (marker.cy - prev_marker.cy) / dt

        prev_marker = marker
        return vx, vy
    
    def detect_rotation_speed(self, prev_marker: TargetDetection, marker: TargetDetection):
        if marker.corners is None or prev_marker is None:
            return None

        axis_now = self.detect_axis(marker)
        axis_prev = self.detect_axis(prev_marker)

        if axis_now is None or axis_prev is None:
            return None

        dt = marker.ts - prev_marker.ts
        if dt <= 0:
            return None

        dtheta = axis_now.x_angle - axis_prev.x_angle
        omega = dtheta / dt

        return omega
