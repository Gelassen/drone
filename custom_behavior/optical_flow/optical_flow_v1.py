import asyncio
import time
import cv2
import numpy as np
import apriltag

from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityNedYaw
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from april_tag_detector import AprilTagDetector
from hardware_interface import HardwareInterface
from drone_hardware import DroneHardware

class AprilTagOpticalController:
    def __init__(self,
                 connection_url="udpin://127.0.0.1:14550",
                 video_source=0,
                 apriltag_detector=AprilTagDetector(),
                 hardware_interface=DroneHardware(),

                 takeoff_alt_m=1.5,
                 vx_limit=0.5,
                 vy_limit=0.5,
                 vz_limit=0.3,
                 Kx=0.4,
                 Ky=0.4,
                 Kz=0.4,
                 run_in_thread_workers=2,

                 ):
        self.connection_url = connection_url
        self.video_source = video_source
        self.takeoff_alt_m = takeoff_alt_m

        self.vx_limit = float(vx_limit)
        self.vy_limit = float(vy_limit)
        self.vz_limit = float(vz_limit)
        self.Kx = float(Kx)
        self.Ky = float(Ky)
        self.Kz = float(Kz)

        self.apriltag_detector = apriltag_detector
        self.hardware_interface = hardware_interface

        # video + executor
        self.cap = None
        self.executor = ThreadPoolExecutor(max_workers=run_in_thread_workers)
        self._run_loop = False

        # frame size populated in open_video()
        self.frame_w = None
        self.frame_h = None

        # last sent velocity time (watchdog)
        self.last_send = 0.0

    # -------------------- MAV helpers --------------------
    async def connect(self):
        await self.hardware_interface.connect()

    async def arm_and_takeoff(self):
        await self.hardware_interface.arm_and_takeoff()

    async def start_offboard(self):
        await self.hardware_interface.start_offboard()

    async def stop_offboard(self):
        await self.hardware_interface.stop_offboard()

    async def land(self):
        await self.hardware_interface.land()

    async def send_velocity(self):
        await self.hardware_interface.send_velocity()

    # video helpers (encapsulating in different class has overhead on managing threads contexts!)
    def open_video(self):
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.video_source}")
        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Opened video {self.video_source} size={self.frame_w}x{self.frame_h}")

    async def read_frame_async(self):
        loop = asyncio.get_running_loop()
        # use lambda wrapper to avoid passing bound C method directly
        ret, frame = await loop.run_in_executor(self.executor, lambda: self.cap.read())
        return ret, frame

    async def get_frame(self):
        ret, frame = await self.read_frame_async()
        return frame if ret and frame is not None else None
    
    # -------------------- Control --------------------
    def compute_velocity_command(self, cx, cy, px_size, target_alt_m):
        """
        Returns vx, vy, vz (down positive for NED VelocityNedYaw) and estimated distance.
        """
        if self.frame_w is None or self.frame_h is None:
            return 0.0, 0.0, 0.0, target_alt_m

        center_x = self.frame_w / 2.0
        center_y = self.frame_h / 2.0
        dx = cx - center_x
        dy = cy - center_y

        px_size = max(float(px_size), 1.0)
        dist = self.apriltag_detector.estimate_distance_from_px(px_size)
        if dist is None:
            dist = target_alt_m

        err_x_m = (dx * dist) / self.apriltag_detector.get_focal_length_px()
        err_y_m = (dy * dist) / self.apriltag_detector.get_focal_length_px()

        # Map to NED velocities: vx -> north (forward), vy -> east (right)
        vx = -self.Ky * err_y_m
        vy = -self.Kx * err_x_m
        # down positive for NED: positive down means moving down, so control as (dist - target)
        vz = self.Kz * (dist - target_alt_m)

        # clamp
        vx = float(max(-self.vx_limit, min(self.vx_limit, vx)))
        vy = float(max(-self.vy_limit, min(self.vy_limit, vy)))
        vz = float(max(-self.vz_limit, min(self.vz_limit, vz)))

        return vx, vy, vz, dist

    # Helper to send velocity safely (checks connection + catches)
    async def send_velocity_safe(self, vx, vy, vz, yaw=0.0):
        isSuccess = await self.hardware_interface.send_velocity(float(vx), float(vy), float(vz), float(yaw))
        if isSuccess:
            self.last_send = time.time()

        return isSuccess

    async def _setup(self, target_alt_m):
        self.open_video()
        await self.connect()

        if not await self.hardware_interface.can_arm_with_backoff():
            raise RuntimeError("Failed to arm")

        await self.arm_and_takeoff()
        await self.start_offboard()

        self.last_send = time.time()
    
    async def _teardown(self):
        print("Stopping loop, landing/cleanup...")

        if self.hardware_interface.is_connected():
            try:
                await self.stop_offboard()
            except Exception:
                pass

            try:
                await self.hardware_interface.land()
            except Exception:
                pass

            try:
                self.executor.shutdown(wait=False)
            except Exception:
                pass

        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def _should_continue(self, start_time, runtime_sec):
        if not self._run_loop:
            return False
        if time.time() - start_time >= runtime_sec:
            return False
        return True

    async def _run_iteration(self, target_alt_m):
        frame = await self.get_frame()
        if frame is None:
            await self._handle_frame_loss()
            return

        detection = self.apriltag_detector.detect_best_target(frame)
        await self._control_step(detection, target_alt_m)

        self._draw_debug(frame, detection)
        self._display_frame(frame)

        await asyncio.sleep(0.001)

    async def _handle_frame_loss(self):
        if time.time() - self.last_send > 0.2:
            await self.send_velocity_safe(0.0, 0.0, 0.0)
            if self.hardware_interface.is_connected():
                try:
                    await self.hardware_interface.land()
                except Exception as e:
                    print(f"Warning: land() failed: {e}")
        await asyncio.sleep(0.02)

    async def _control_step(self, detection, target_alt_m):
        if detection is None:
            await self._hover_if_needed()
            return

        vx, vy, vz, _ = self.compute_velocity_command(
            detection.cx,
            detection.cy,
            detection.px_size,
            target_alt_m
        )

        await self.send_velocity_safe(vx, vy, vz)

    async def _hover_if_needed(self):
        if time.time() - self.last_send > 0.2:
            await self.send_velocity_safe(0.0, 0.0, 0.0)

    def _draw_debug(self, frame, detection):
        if detection:
            cv2.circle(frame, (int(detection.cx), int(detection.cy)), 6, (0, 0, 255), -1)

    def _display_frame(self, frame):
        try:
            cv2.imshow("AprilTag Controller", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self._run_loop = False
        except Exception:
            pass


    # -------------------- Main loop --------------------
    async def run(self, runtime_sec=120.0, target_alt_m=None):
        target_alt_m = target_alt_m or self.takeoff_alt_m

        await self._setup(target_alt_m)

        self._run_loop = True
        start_time = time.time()

        try:
            while self._should_continue(start_time, runtime_sec):
                await self._run_iteration(target_alt_m)
        finally:
            await self._teardown()

async def main():
    controller = AprilTagOpticalController(
        connection_url="udpin://127.0.0.1:14550",
        video_source="../../assets/ar_test_video.MOV",

        takeoff_alt_m=1.5,

        vx_limit=0.5,
        vy_limit=0.5,
        vz_limit=0.3,

        Kx=0.4,
        Ky=0.4,
        Kz=0.4,

        run_in_thread_workers=2,
    )

    await controller.run(runtime_sec=120.0, target_alt_m=1.5)


if __name__ == "__main__":
    asyncio.run(main())
