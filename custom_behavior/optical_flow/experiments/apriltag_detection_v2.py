import asyncio
import cv2
import numpy as np
import apriltag
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityNedYaw


# ----------------------------------
# AprilTag detector (unchanged)
# ----------------------------------
apriltag_detector = apriltag.Detector(
    apriltag.DetectorOptions(families="tag16h5")
)

MIN_AREA = 2000
MAX_AREA = 15000
ASPECT_RATIO_TOL = 0.2


def find_squares(frame):
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
            aspect_ratio = w / h
            if MIN_AREA <= area <= MAX_AREA and abs(aspect_ratio - 1) < ASPECT_RATIO_TOL:
                squares.append((x, y, w, h))
    return squares


# ----------------------------------
# Controller parameters
# ----------------------------------
TAG_SIZE_M   = 0.07
FOCAL_PX     = 700.0
KZ           = 0.4
VZ_LIM       = 0.3

# reference: tag at 1.5 m should be approx this many pixels
TAG_REF_PX   = 120  # adjust experimentally for your camera/altitude


def estimate_height_from_tag(px_size):
    # simple stable estimator
    if px_size <= 1:
        return None
    return TAG_REF_PX / px_size * 1.5   # reference altitude 1.5 m


# ----------------------------------
# Main controller loop
# ----------------------------------
async def main():
    drone = System()
    print("Connecting to udpin://127.0.0.1:14550...")
    await drone.connect(system_address="udp://127.0.0.1:14550")

    print("Waiting for drone...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Drone connected!")
            break

    print("Arming…")
    await drone.action.arm()

    print("Takeoff to 1.5 m…")
    await drone.action.takeoff()
    await asyncio.sleep(5)

    print("Starting offboard…")
    try:
        await drone.offboard.start()
    except OffboardError as e:
        print("Offboard start failed:", e)
        return

    print("Offboard started.")

    cap = cv2.VideoCapture("../../assets/ar_test_video.MOV")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video ended, landing…")
            break

        squares = find_squares(frame)
        tag_px = None

        for (x, y, w, h) in squares:
            roi = frame[y:y+h, x:x+w]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            detections = apriltag_detector.detect(gray_roi)

            if len(detections) == 1:
                det = detections[0]
                # bounding box size as scale proxy
                tag_px = max(w, h)

        # ----------------------------------
        # HEIGHT CONTROL
        # ----------------------------------
        if tag_px is None:
            vz = 0    # freeze height if no tag
            print("No tag → vz=0")
        else:
            h_est = estimate_height_from_tag(tag_px)
            if h_est is None:
                vz = 0
            else:
                # maintain 1.5 m target
                dz = h_est - 1.5
                vz = np.clip(KZ * dz, -VZ_LIM, VZ_LIM)
                print(f"Tag px={tag_px}, h_est={h_est:.2f}, vz={vz:.2f}")

        # ----------------------------------
        # Send velocity to robot
        # ----------------------------------
        try:
            await drone.offboard.set_velocity_ned(
                VelocityNedYaw(vx=0.0, vy=0.0, vz=-vz, yaw=0.0)
            )
        except OffboardError as e:
            print("Offboard error:", e)
            break

        await asyncio.sleep(0.05)

    await drone.action.land()
    await asyncio.sleep(3)
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
