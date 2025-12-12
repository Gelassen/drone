import numpy as np

from .tag_geometry import TagGeometry
from .tag_selector import TagSelector


class DroneController:
    def __init__(
        self,
        detector,
        hardware,
        selector=None,
        focal_length_px=700,
        tag_size_m=0.07,
        angle_cos_threshold=0.7,
        Kx=0.6, Ky=0.6, Kz=0.8,
        vx_limit=0.3, vy_limit=0.3, vz_limit=0.25,
        is_debug=True,
    ):
        self.detector = detector
        self.selector = selector or TagSelector()
        self.hardware = hardware

        self.focal_length_px = focal_length_px
        self.tag_size_m = tag_size_m
        self.angle_cos_threshold = angle_cos_threshold

        self.Kx = Kx; self.Ky = Ky; self.Kz = Kz
        self.vx_limit = vx_limit; self.vy_limit = vy_limit; self.vz_limit = vz_limit

        self.IS_DEBUG = is_debug

    def log(self, *a):
        if self.IS_DEBUG:
            print(*a)

    async def process_frame(self, gray, target_alt_m):
        print("[start] drone_controller::process_frame")
        camera_params = [
            self.focal_length_px,
            self.focal_length_px,
            gray.shape[1] / 2,
            gray.shape[0] / 2,
        ]

        tags = self.detector.detect(gray, camera_params, self.tag_size_m)
        tag = self.selector.select_best(tags)

        if not tag:
            return (0.0, 0.0, 0.0)

        px = TagGeometry.px_size_from_corners(tag.corners)
        cos_z = TagGeometry.pose_cos_z(tag.pose_R)

        if cos_z is None or cos_z < self.angle_cos_threshold:
            return (0, 0, 0)

        h_px = TagGeometry.height_from_px(
            px, cos_z, self.focal_length_px, self.tag_size_m
        )
        h_pose = TagGeometry.pose_height(tag.pose_t)
        height = h_pose or h_px

        cx = float(tag.corners[:, 0].mean())
        cy = float(tag.corners[:, 1].mean())
        img_cx = gray.shape[1] / 2
        img_cy = gray.shape[0] / 2

        err_x = (cx - img_cx) * height / self.focal_length_px
        err_y = (cy - img_cy) * height / self.focal_length_px

        vx = float(np.clip(-self.Ky * err_y, -self.vx_limit, self.vx_limit))
        vy = float(np.clip(-self.Kx * err_x, -self.vy_limit, self.vy_limit))
        vz = float(np.clip(self.Kz * (height - target_alt_m),
                           -self.vz_limit, self.vz_limit))

        await self._send_velocity(vx, vy, vz)
        return vx, vy, vz

    async def _send_velocity(self, vx, vy, vz):
        await self.hardware.send_velocity(vx, vy, vz)
