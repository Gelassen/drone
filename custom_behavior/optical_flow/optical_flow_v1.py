import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import apriltag
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityNedYaw


class AprilTagOpticalController:
    def __init__(self,
                 connection_url="udpin://127.0.0.1:14550",
                 video_source=0,
                 tag_family="tag16h5",
                 tag_size_m=0.05,
                 focal_length_px=600.0,
                 takeoff_alt_m=1.5,
                 vx_limit=0.5,
                 vy_limit=0.5,
                 vz_limit=0.3,
                 Kx=0.4,
                 Ky=0.4,
                 Kz=0.4,
                 run_in_thread_workers=2,
                 disable_mav=False):
        self.connection_url = connection_url
        self.video_source = video_source
        self.tag_size_m = tag_size_m
        self.focal_length_px = focal_length_px
        self.takeoff_alt_m = takeoff_alt_m

        self.vx_limit = float(vx_limit)
        self.vy_limit = float(vy_limit)
        self.vz_limit = float(vz_limit)
        self.Kx = float(Kx)
        self.Ky = float(Ky)
        self.Kz = float(Kz)

        self._connected = False
        self.disable_mav = disable_mav

        # mavsdk system (may be None in disable_mav mode)
        self.drone = System() if not disable_mav else None

        # apriltag detector
        self.at_options = apriltag.DetectorOptions(families=tag_family)
        self.apriltag_detector = apriltag.Detector(self.at_options)

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
        if self.disable_mav:
            print("MAV disabled, skipping connect")
            return
        print("Connecting to drone...")
        try:
            await self.drone.connect(system_address=self.connection_url)
        except Exception as e:
            # connect may raise if bad URL; continue but mark disconnected
            print(f"Warning: drone.connect() exception: {e}")
            self._connected = False
            return

        try:
            async for state in self.drone.core.connection_state():
                if state.is_connected:
                    print("Drone discovered!")
                    self._connected = True
                    break
        except Exception as e:
            print(f"Warning: connection_state() exception: {e}")
            self._connected = False

    async def arm_and_takeoff(self):
        if self.disable_mav or not self._connected:
            print("Skipping arm/takeoff (disabled or not connected)")
            return
        print("Arming...")
        try:
            await self.drone.action.arm()
            print(f"Taking off to {self.takeoff_alt_m} m...")
            await self.drone.action.takeoff()
            async for pos in self.drone.telemetry.position():
                if pos.relative_altitude_m >= self.takeoff_alt_m * 0.95:
                    print("Reached target altitude")
                    break
                await asyncio.sleep(0.2)
        except Exception as e:
            print(f"Warning: arm/takeoff failed: {e}")
            # don't raise — continue in safe mode
            self._connected = False

    async def start_offboard(self):
        if self.disable_mav or not self._connected:
            return
        try:
            await self.drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0, 0, 0))
            await self.drone.offboard.start()
            print("Offboard started")
        except OffboardError as e:
            print(f"Failed to start Offboard: {e._result.result}")
            raise
        except Exception as e:
            print(f"Warning: start_offboard failed: {e}")
            self._connected = False

    async def stop_offboard(self):
        if self.disable_mav or not self._connected:
            return
        try:
            await self.drone.offboard.stop()
            print("Offboard stopped")
        except Exception as e:
            print(f"Warning: stop_offboard failed: {e}")

    # -------------------- Video helpers --------------------
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

    # -------------------- Detection --------------------
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
        dist = self.estimate_distance_from_px(px_size)
        if dist is None:
            dist = target_alt_m

        err_x_m = (dx * dist) / self.focal_length_px
        err_y_m = (dy * dist) / self.focal_length_px

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
        """
        Send velocity if connected; otherwise no-op.
        Keeps last_send updated only on success.
        """
        if not self._connected or self.disable_mav:
            return False
        try:
            await self.drone.offboard.set_velocity_ned(VelocityNedYaw(float(vx), float(vy), float(vz), float(yaw)))
            self.last_send = time.time()
            return True
        except Exception as e:
            print(f"Warning: set_velocity_ned failed: {e}")
            return False

    # -------------------- Main loop --------------------
    async def run(self, runtime_sec=120.0, target_alt_m=None):
        if target_alt_m is None:
            target_alt_m = self.takeoff_alt_m

        self.open_video()
        await self.connect()
        await self.arm_and_takeoff()
        await self.start_offboard()

        self._run_loop = True
        start_time = time.time()
        lost_frames = 0

        try:
            while self._run_loop and (time.time() - start_time) < runtime_sec:
                ret, frame = await self.read_frame_async()
                if not ret or frame is None:
                    lost_frames += 1
                    # if video is lost for multiple frames, attempt safe landing when connected
                    if lost_frames > 3:
                        print("VIDEO LOST → performing safe hover/land")
                        # send immediate zero velocity and land if connected
                        await self.send_velocity_safe(0.0, 0.0, 0.0)
                        if self._connected:
                            try:
                                await self.drone.action.land()
                            except Exception as e:
                                print(f"Warning: land() failed: {e}")
                        break
                    await asyncio.sleep(0.02)
                    continue

                # reset lost frames counter once we have a frame
                lost_frames = 0

                # --- detection ---
                squares = self.find_squares(frame)
                best_cx = best_cy = best_px_size = None
                best_tag = None

                # try to detect real tags first
                for (x, y, w, h) in squares:
                    # defensive cropping: ensure ROI in frame bounds
                    x0 = max(0, x)
                    y0 = max(0, y)
                    x1 = min(self.frame_w, x + w)
                    y1 = min(self.frame_h, y + h)
                    if x1 <= x0 or y1 <= y0:
                        continue
                    roi = frame[y0:y1, x0:x1]

                    try:
                        tags = self.detect_tags_in_roi(roi)
                    except Exception as e:
                        # detector occasionally throws; skip ROI
                        print(f"Warning: detect_tags_in_roi failed: {e}")
                        tags = []

                    if not tags:
                        continue

                    for t in tags:
                        corners = np.array(t.corners, dtype=float)
                        side1 = float(np.linalg.norm(corners[0] - corners[1]))
                        side2 = float(np.linalg.norm(corners[1] - corners[2]))
                        side = 0.5 * (side1 + side2)

                        cx = float(corners[:, 0].mean() + x0)
                        cy = float(corners[:, 1].mean() + y0)

                        # safe comparison with None
                        if best_px_size is None or side > best_px_size:
                            best_tag = t
                            best_px_size = side
                            best_cx = cx
                            best_cy = cy

                # fallback: use largest square candidate if no real tag
                if best_tag is None and squares:
                    # select largest by area
                    x, y, w, h = max(squares, key=lambda s: s[2] * s[3])
                    best_cx = x + w / 2.0
                    best_cy = y + h / 2.0
                    best_px_size = float(max(w, h))

                # --- compute & send velocity ---
                if best_cx is not None:
                    vx, vy, vz, est_dist = self.compute_velocity_command(best_cx, best_cy, best_px_size, target_alt_m)
                    # send only if connected; otherwise we just draw/debug
                    await self.send_velocity_safe(vx, vy, vz, 0.0)

                    # draw debug overlays
                    try:
                        cv2.circle(frame, (int(best_cx), int(best_cy)), 6, (0, 0, 255), -1)
                        cv2.putText(frame, f"vx:{vx:.2f} vy:{vy:.2f} vz:{vz:.2f} dist:{est_dist:.2f}",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    except Exception:
                        pass
                else:
                    # NO candidate: hover/stop (behavior A)
                    # ensure we periodically send zero velocities so autopilot doesn't keep old command
                    if time.time() - self.last_send > 0.2:
                        await self.send_velocity_safe(0.0, 0.0, 0.0)

                # draw candidate squares
                for x, y, w, h in squares:
                    try:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
                    except Exception:
                        pass

                # show frame (if display available)
                try:
                    cv2.imshow("AprilTag Optical Controller", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("User requested exit (q).")
                        break
                except Exception:
                    # headless or imshow error: ignore
                    pass

                # small sleep to yield control (keeps loop responsive)
                await asyncio.sleep(0.001)

        finally:
            print("Stopping loop, landing/cleanup...")
            # stop offboard and land if connected
            if self._connected:
                try:
                    await self.stop_offboard()
                except Exception:
                    pass
                try:
                    await self.drone.action.land()
                except Exception:
                    pass

            # cleanup video + executor
            try:
                if self.cap is not None:
                    self.cap.release()
            except Exception:
                pass
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    async def shutdown(self):
        self._run_loop = False
        try:
            self.executor.shutdown(wait=False)
        except Exception:
            pass
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


# -------------------- Example usage --------------------
async def main():
    controller = AprilTagOpticalController(
        connection_url="udpin://127.0.0.1:14550",
        video_source="../../assets/ar_test_video.MOV",

        tag_family="tag16h5",
        tag_size_m=0.07,
        focal_length_px=700.0,

        takeoff_alt_m=1.5,

        vx_limit=0.5,
        vy_limit=0.5,
        vz_limit=0.3,

        Kx=0.4,
        Ky=0.4,
        Kz=0.4,

        run_in_thread_workers=2,
        disable_mav=False
    )

    await controller.run(runtime_sec=120.0, target_alt_m=1.5)


if __name__ == "__main__":
    asyncio.run(main())
