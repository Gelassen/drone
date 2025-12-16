import numpy as np

class OpticalVelocityController:
    def __init__(self, detector, frame_w, frame_h,
                 Kx, Ky, Kz, vx_lim, vy_lim, vz_lim, dead_px):
        self.detector = detector
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.Kx = Kx
        self.Ky = Ky
        self.Kz = Kz
        self.vx_lim = vx_lim
        self.vy_lim = vy_lim
        self.vz_lim = vz_lim
        self.dead_px = dead_px

    def compute(self, detection, target_alt):
        cx, cy, px_size = detection.cx, detection.cy, detection.px_size

        dx = cx - self.frame_w / 2
        dy = cy - self.frame_h / 2

        if abs(dx) < self.dead_px:
            dx = 0
        if abs(dy) < self.dead_px:
            dy = 0

        px_size = max(px_size, 1.0)
        dist = self.detector.estimate_distance_from_px(px_size) or target_alt

        focal = self.detector.get_focal_length_px()
        err_x = (dx * dist) / focal
        err_y = (dy * dist) / focal

        vx = -self.Ky * err_y
        vy = -self.Kx * err_x
        vz = self.Kz * (dist - target_alt)

        return (
            float(np.clip(vx, -self.vx_lim, self.vx_lim)),
            float(np.clip(vy, -self.vy_lim, self.vy_lim)),
            float(np.clip(vz, -self.vz_lim, self.vz_lim)),
            dist
        )
