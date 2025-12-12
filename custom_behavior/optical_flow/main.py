import os
import cv2
import asyncio
import numpy as np

from .drone_controller import DroneController
from .dependencies import provide_detector, provide_hardware

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "../../assets/ar_test_video.MOV")

print(cv2.getBuildInformation())

async def main():
    print("[start] main")
    cap = cv2.VideoCapture(VIDEO_PATH)

    # вручную внедряем зависимости
    detector = provide_detector()
    hardware = provide_hardware()

    controller = DroneController(
        detector=detector,
        hardware=hardware,
        is_debug=True
    )

    if not cap.isOpened():
        print("ERROR: Cannot open video file")
        return

    print("Backend:", cap.getBackendName())

    while True:
        ok, frame = cap.read()

        if not ok:
            break        

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        await controller.process_frame(gray, target_alt_m=1.0)

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # while True:
    #     ok, frame = cap.read()
    #     if not ok or frame is None:
    #         print("Something went wrong while reading the frame")
    #         print("ok:", ok, "frame is None:", frame is None)
    #         break

    #     print("shape:", frame.shape)
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     await controller.process_frame(gray, target_alt_m=1.0)

    #     if cv2.waitKey(1) == 27:
    #         break

    cap.release()
    cv2.destroyAllWindows()

    print("[end] main")


if __name__ == "__main__":
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.imshow("test", img)
    cv2.waitKey(0)
    # asyncio.run(main())
