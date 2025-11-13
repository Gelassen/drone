import signal, sys
import numpy as np
import cv2
from aruco_detector import ArUcoDetector

def main():
    # Псевдокалибровка (подбери под своё видео/камеру)
    camera_matrix = np.array([[920, 0, 640],
                              [0, 920, 360],
                              [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((5,1), dtype=np.float32)
    marker_length = 0.10  # метры (пример)

    detector = ArUcoDetector(camera_matrix, dist_coeffs, marker_length, dictionary=cv2.aruco.DICT_5X5_50, alpha=0.45)

    # Источник: видеофайл или 0 для вебкамеры
    cap = cv2.VideoCapture("../../assets/ar_test_video.MOV")

    if not cap.isOpened():
        print("❌ Не удалось открыть источник видео.")
        return

    def handle_exit(*args):
        cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)

    print("✅ Видео открыто. Нажмите ESC для выхода. Наблюдаемые логи печатаются в консоль.")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Конец видео или ошибка чтения.")
            break

        frame_idx += 1
        corners, ids, rejected = detector.detect(frame)
        frame_out, poses = detector.postProcess(frame, corners, ids, rejected, debug=True)

        # вывод в консоль для диагностики
        det_count = 0 if ids is None else len(ids)
        rej_count = 0 if rejected is None else len(rejected)
        pose_count = len(poses)
        print(f"[frame {frame_idx}] detected={det_count}, rejected={rej_count}, poses={pose_count}", end='')
        if pose_count > 0:
            print(" ids:", [p[0] for p in poses])
            for pid, r, t in poses:
                # печатаем угол/позицию для человека: r — Rodrigues vector (3x1), t — (3x1)
                try:
                    R_mat, _ = cv2.Rodrigues(r)
                    roll = np.degrees(np.arctan2(R_mat[2,1], R_mat[2,2]))
                    pitch = np.degrees(np.arctan2(-R_mat[2,0], np.sqrt(R_mat[2,1]**2 + R_mat[2,2]**2)))
                    yaw = np.degrees(np.arctan2(R_mat[1,0], R_mat[0,0]))
                    print(f"  id={pid} t=[{t.ravel()[0]:.3f},{t.ravel()[1]:.3f},{t.ravel()[2]:.3f}] m rpy=[{roll:.1f},{pitch:.1f},{yaw:.1f}]°")
                except Exception:
                    print(f"  id={pid} pose_print_failed")

        else:
            print()

        cv2.imshow("ArUco Detector", frame_out)
        key = cv2.waitKey(10) & 0xFF
        if key == 27 or cv2.getWindowProperty("ArUco Detector", cv2.WND_PROP_VISIBLE) < 1:
            handle_exit()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
