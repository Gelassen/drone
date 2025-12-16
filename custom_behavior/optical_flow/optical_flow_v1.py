
import asyncio
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
from target_detection import TargetDetection

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


if __name__ == "__main__":
    asyncio.run(main())
