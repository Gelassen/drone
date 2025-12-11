#!/usr/bin/env python3
"""
Refactored AprilTag optical controller for testable design.

- Detector and MAV interface injected
- Pure computation isolated for unit testing
- Logging centralized via IS_DEBUG
"""

import asyncio
import os
import time
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from april_tag_detector import AprilTagDetector
from hardware_interface import HardwareInterface
from drone_hardware import DroneHardware
from optical_velocity_controller import OpticalVelocityController
from target_tracker import TargetTracker
from video_source import AsyncVideoSource

@dataclass
class TargetDetection:
    cx: float
    cy: float
    px_size: float
    source: str  # "tag" | "square"

class AprilTagOpticalController:
    def __init__(
        self,
        video_source,
        hardware: DroneHardware,
        detector=AprilTagDetector(),
        takeoff_alt_m=1.5,
        loop_hz=30,
        lost_frame_threshold=100,
        Kx=0.4, Ky=0.4, Kz=0.4,
        vx_limit=0.5, vy_limit=0.5, vz_limit=0.3,
        dead_px=5,
        executor_workers=2
    ):
        self.loop_dt = 1.0 / loop_hz
        self.takeoff_alt = takeoff_alt_m

        self.hardware = hardware
        self.detector = detector

        self.executor = ThreadPoolExecutor(max_workers=executor_workers)
        self.video = AsyncVideoSource(video_source, self.executor)
        self.tracker = TargetTracker(detector, lost_frame_threshold)

        self.velocity_ctrl = None

        self.dead_px = dead_px
        self.Kx, self.Ky, self.Kz = Kx, Ky, Kz
        self.vx_limit, self.vy_limit, self.vz_limit = vx_limit, vy_limit, vz_limit

        self._run = False
        self.last_send = 0.0

    # ---------- Lifecycle ----------

    async def setup(self):
        await self.video.open()
        await self.hardware.connect()

        if not await self.hardware.can_arm_with_backoff():
            raise RuntimeError("Arm failed")

        await self.hardware.arm_and_takeoff()
        await self.hardware.start_offboard()

        self.velocity_ctrl = OpticalVelocityController(
            self.detector,
            self.video.frame_w,
            self.video.frame_h,
            self.Kx, self.Ky, self.Kz,
            self.vx_limit, self.vy_limit, self.vz_limit,
            self.dead_px
        )

        self.last_send = time.time()

    async def teardown(self):
        print("[Controller] Shutdown")

        if self.hardware.is_connected():
            try:
                await self.hardware.stop_offboard()
                await self.hardware.land()
            except Exception:
                pass

        self.video.close()
        cv2.destroyAllWindows()

        self.executor.shutdown(wait=True, cancel_futures=True)

    # ---------- Control ----------

    async def _send_velocity_safe(self, vx, vy, vz):
        ok = await self.hardware.send_velocity(vx, vy, vz, 0.0)
        if ok:
            self.last_send = time.time()

    async def _hover_if_needed(self):
        if time.time() - self.last_send > 0.2:
            await self._send_velocity_safe(0.0, 0.0, 0.0)

    async def _on_frame_lost(self):
        await self._hover_if_needed()
        await asyncio.sleep(0.02)

    # ---------- Loop ----------

    async def _iteration(self, target_alt):
        frame = await self.video.read()
        if frame is None:
            await self._on_frame_lost()
            return

        detection = self.tracker.process(frame)

        if self.tracker.is_lost():
            print("[Safety] Target lost → landing")
            await self.hardware.land()
            self._run = False
            return

        if detection:
            vx, vy, vz, _ = self.velocity_ctrl.compute(detection, target_alt)
            await self._send_velocity_safe(vx, vy, vz)
        else:
            await self._hover_if_needed()

        self._draw_debug(frame, detection)
        self._display(frame)

    # ---------- UI ----------

    def _draw_debug(self, frame, detection):
        if detection:
            cv2.circle(frame, (int(detection.cx), int(detection.cy)), 6, (0, 0, 255), -1)

    def _display(self, frame):
        cv2.imshow("AprilTag Controller", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self._run = False

    # ---------- Public ----------

    async def run(self, runtime_sec=120.0, target_alt=None):
        target_alt = target_alt or self.takeoff_alt
        await self.setup()

        self._run = True
        start = time.time()

        try:
            while self._run and time.time() - start < runtime_sec:
                t0 = time.monotonic()
                await self._iteration(target_alt)
                dt = time.monotonic() - t0
                await asyncio.sleep(max(0.0, self.loop_dt - dt))
        finally:
            await self.teardown()


# ===================== Entry =====================

async def main():
    controller = AprilTagOpticalController(
        video_source="../../assets/ar_test_video.MOV",
        hardware=DroneHardware(),
    )
    await controller.run(runtime_sec=120.0)

    def _pose_height(self, pose_t):
        try:
            return abs(float(pose_t[2, 0]))
        except Exception:
            try:
                return abs(float(pose_t[2]))
            except Exception:
                return None

    def _pose_cos_z(self, pose_R):
        try:
            normal = pose_R[:, 2]
            return abs(float(normal[2]))
        except Exception:
            return None

    def _height_from_px(self, px_size, cos_theta):
        if px_size is None:
            return None
        if cos_theta is None:
            cos_theta = 1.0
        if cos_theta <= 0.01:
            return None
        corrected = px_size * cos_theta
        if corrected < 1.0:
            return None
        return float((self.focal_length_px * self.tag_size_m) / corrected)

    def _median_height(self, value):
        if value is None:
            return None
        self.height_hist.append(value)
        return float(np.median(self.height_hist))

    def _estimate_height(self, tag):
        if tag is None:
            return None, None, None
        raw_px = self._px_size_from_corners(tag.corners)
        self.filtered_px = self._lowpass(self.filtered_px, raw_px, alpha=self.filter_alpha)
        cos_z = self._pose_cos_z(tag.pose_R)
        if cos_z is None or cos_z < self.angle_cos_threshold:
            self.log(f"Tag too angled (cos_z={cos_z}) -> skip height")
            return None, cos_z, raw_px
        h_pose = self._pose_height(tag.pose_t)
        h_px = self._height_from_px(self.filtered_px, cos_z)
        height = h_pose if h_pose is not None and 0.02 < h_pose < 50.0 else h_px
        if height is not None:
            height_med = self._median_height(height)
            self.filtered_height = self._lowpass(self.filtered_height, height_med, alpha=0.4)
            return self.filtered_height, cos_z, raw_px
        return None, cos_z, raw_px

    def _compute_velocity_commands(self, tag, height_used, target_alt_m):
        vx_cmd = vy_cmd = vz_cmd = 0.0
        if tag is None or height_used is None:
            return vx_cmd, vy_cmd, vz_cmd
        # pixel center
        cx = float(np.mean(tag.corners[:, 0]))
        cy = float(np.mean(tag.corners[:, 1]))
        img_cx = np.mean([tag.corners[:, 0].min(), tag.corners[:, 0].max()])
        img_cy = np.mean([tag.corners[:, 1].min(), tag.corners[:, 1].max()])
        dx_px = cx - img_cx
        dy_px = cy - img_cy
        # convert pixel errors -> meters using estimated height
        err_x_m = (dx_px * height_used) / self.focal_length_px
        err_y_m = (dy_px * height_used) / self.focal_length_px
        # NED mapping
        vx_raw = -self.Ky * err_y_m
        vy_raw = -self.Kx * err_x_m
        vz_raw = self.Kz * (height_used - target_alt_m)
        vx_cmd = float(np.clip(vx_raw, -self.vx_limit, self.vx_limit))
        vy_cmd = float(np.clip(vy_raw, -self.vy_limit, self.vy_limit))
        vz_cmd = float(np.clip(vz_raw, -self.vz_limit, self.vz_limit))
        # filtering
        self.filtered_vx = self._lowpass(self.filtered_vx, vx_cmd, alpha=0.25)
        self.filtered_vy = self._lowpass(self.filtered_vy, vy_cmd, alpha=0.25)
        self.filtered_vz = self._lowpass(self.filtered_vz, vz_cmd, alpha=0.25)
        return self.filtered_vx, self.filtered_vy, self.filtered_vz

    # ---------- detection ----------
    def _detect_tags(self, gray_frame):
        camera_params = [
            self.focal_length_px,
            self.focal_length_px,
            gray_frame.shape[1] / 2.0,
            gray_frame.shape[0] / 2.0,
        ]
        tags = self.detector.detect(gray_frame, camera_params=camera_params, tag_size=self.tag_size_m)
        self.log(f"Detected {len(tags)} tags")
        return tags

    def _select_best_tag(self, tags):
        if not tags:
            return None
        return max(tags, key=lambda t: self._px_size_from_corners(t.corners))

    # ---------- send / MAV commands ----------
    async def _send_velocity(self, vx, vy, vz):
        await self.hardware.send_velocity(vx, vy, vz)

# -------------------- simple usage example --------------------
if __name__ == "__main__":
    asyncio.run(main())
