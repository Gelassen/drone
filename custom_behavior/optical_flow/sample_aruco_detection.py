import cv2
import numpy as np
import signal
import sys

# --- 1. Список словарей для перебора ---
ARUCO_DICTS = [
    cv2.aruco.DICT_4X4_50,
    cv2.aruco.DICT_4X4_100,
    cv2.aruco.DICT_5X5_50,
]

# --- 2. Подготовка параметров ---
try:
    parameters = cv2.aruco.DetectorParameters()
    print("OpenCV >= 4.7 is used")
except AttributeError:
    parameters = cv2.aruco.DetectorParameters_create()
    print("OpenCV < 4.7 (legacy) is used")

# --- 3. Псевдокалибровка камеры ---
camera_matrix = np.array([[920, 0, 640],
                          [0, 920, 360],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1))
marker_length = 0.1  # метры

# --- 4. Видео ---
cap = cv2.VideoCapture("../../assets/ar_test_video.MOV")
if not cap.isOpened():
    print("❌ Не удалось открыть видео.")
    sys.exit(1)
print("✅ Видео открыто. Нажмите ESC или закройте окно для выхода.")

# --- 5. Обработчик завершения ---
def handle_exit(*args):
    print("\n🛑 Завершение работы...")
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)

# --- 6. Безопасная функция отрисовки осей ---
def draw_axes_safe(frame, camera_matrix, dist_coeffs, rvec, tvec, length=0.05):
    """Рисует 3D-оси вручную, если drawFrameAxes недоступна"""
    try:
        if hasattr(cv2, "drawFrameAxes"):
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, length)
            return
        elif hasattr(cv2.aruco, "drawAxis"):
            cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, length)
            return
    except Exception:
        pass

    # Ручная отрисовка
    R, _ = cv2.Rodrigues(rvec)
    axis_len = length
    axis_points = np.float32([
        [0, 0, 0],
        [axis_len, 0, 0],
        [0, axis_len, 0],
        [0, 0, axis_len]
    ])
    proj, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
    proj = proj.reshape(-1, 2).astype(int)
    cv2.line(frame, tuple(proj[0]), tuple(proj[1]), (0, 0, 255), 2)  # X (красный)
    cv2.line(frame, tuple(proj[0]), tuple(proj[2]), (0, 255, 0), 2)  # Y (зелёный)
    cv2.line(frame, tuple(proj[0]), tuple(proj[3]), (255, 0, 0), 2)  # Z (синий)

# --- 7. Главный цикл ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Конец видео или ошибка чтения.")
        break

    detected = False

    # Пробуем все словари
    for dict_type in ARUCO_DICTS:
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
        parameters = cv2.aruco.DetectorParameters()
        parameters.adaptiveThreshConstant = 5  # вместо 7–10
        parameters.minMarkerPerimeterRate = 0.02  # уменьши для дальних
        parameters.maxMarkerPerimeterRate = 4.0
        parameters.polygonalApproxAccuracyRate = 0.02

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # gray = cv2.equalizeHist(gray)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (3,3), 0)

        corners, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # gray = cv2.equalizeHist(gray)
        # gray = cv2.GaussianBlur(gray, (3,3), 0)
        # corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        # if len(rejected) > 0:
        #     cv2.aruco.drawDetectedMarkers(frame, rejected, borderColor=(128, 0, 128))


        if ids is not None and len(ids) > 0:
            detected = True
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, marker_length, camera_matrix, dist_coeffs
            )
            if True:
                # draw axis
                for rvec, tvec in zip(rvecs, tvecs):
                    draw_axes_safe(frame, camera_matrix, dist_coeffs, rvec, tvec)

                    # Расчёт углов Эйлера
                    R, _ = cv2.Rodrigues(rvec)
                    roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
                    pitch = np.degrees(np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2)))
                    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))

                    cv2.putText(frame, f"R:{roll:6.1f} P:{pitch:6.1f} Y:{yaw:6.1f}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    print(f"[DICT_{dict_type}] Roll: {roll:7.2f}°, Pitch: {pitch:7.2f}°, Yaw: {yaw:7.2f}°")
            else:
                # light up not detected 
                # Подсветить маркеры, у которых нет позы
                c = corners[i][0]
                cv2.polylines(frame, [c.astype(int)], True, (0, 0, 255), 2)


            break

    if not detected:
        cv2.putText(frame, "Marker not detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("ArUco Pose Estimation", frame)

    # --- Завершение по ESC или закрытию окна ---
    key = cv2.waitKey(10) & 0xFF
    if key == 27 or cv2.getWindowProperty("ArUco Pose Estimation", cv2.WND_PROP_VISIBLE) < 1:
        handle_exit()
