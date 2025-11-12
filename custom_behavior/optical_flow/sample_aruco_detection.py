import signal, sys
import numpy as np
import cv2
from aruco_detector import ArUcoDetector

camera_matrix = np.array([[920, 0, 640],
                          [0, 920, 360],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((5,1))
marker_length = 0.1

detector = ArUcoDetector(camera_matrix, dist_coeffs, marker_length)

cap = cv2.VideoCapture("../../assets/ar_test_video.MOV")

def handle_exit(*args):
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    corners, ids, rejected = detector.detect(frame)
    frame, poses = detector.postProcess(frame, corners, ids, rejected)

    cv2.imshow("ArUco Detector", frame)
    if cv2.waitKey(10) & 0xFF == 27:
        handle_exit()
