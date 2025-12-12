# import signal, sys
# import numpy as np
# import cv2
# from aruco_detector import (
#     ArUcoDetector,
#     AdvancedArUcoDetector,
#     VideoProcessor
# )

# main.py
import signal
import sys
import numpy as np
import cv2
from new_aruco_detector import (
    AruCoDetector, 
    DetectionStrategy, 
    VideoProcessor
)

def get_improved_camera_calibration():
    """Улучшенная калибровка с учетом дисторсий"""
    camera_matrix = np.array([[920, 0, 640],
                              [0, 920, 360], 
                              [0, 0, 1]], dtype=np.float32)
    
    # Реалистичные коэффициенты дисторсии
    dist_coeffs = np.array([-0.2, 0.1, 0.001, 0.001, 0.0], dtype=np.float32)
    
    return camera_matrix, dist_coeffs

def main():
    """Главная функция приложения"""
    
    ARUCO_DICTS = [
        cv2.aruco.DICT_4X4_50,
        cv2.aruco.DICT_4X4_100,
        cv2.aruco.DICT_5X5_50,
    ]
    
    camera_matrix, dist_coeffs = get_improved_camera_calibration()
    
    def handle_exit(signum=None, frame=None):
        print("\n🛑 Завершение работы...")
        cv2.destroyAllWindows()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, handle_exit)
    
    video_processor = None
    try:
        # Пробуем разные стратегии
        strategies = [
            DetectionStrategy.STANDARD,
            DetectionStrategy.MULTI_PASS, 
            DetectionStrategy.REGION_AWARE,
            DetectionStrategy.ADAPTIVE
        ]
        
        video_processor = VideoProcessor("../../assets/ar_test_video.MOV")
        
        for strategy in strategies:
            print(f"\n🔍 Тестируем стратегию: {strategy.value}")
            
            aruco_detector = AruCoDetector(
                ARUCO_DICTS, camera_matrix, dist_coeffs, strategy=strategy
            )
            
            # Сбрасываем видео
            video_processor.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Тестируем несколько кадров
            for i in range(30):  # 30 кадров для теста
                ret, frame = video_processor.read_frame()
                if not ret:
                    break
                
                # Обработка
                detected = aruco_detector.process(frame)
                aruco_detector.postProcess(frame)
                
                cv2.imshow("ArUco Pose Estimation", frame)
                
                if cv2.waitKey(10) & 0xFF == 27:
                    handle_exit()
                    return
                    
            cv2.waitKey(1000)  # Пауза между стратегиями
                
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    finally:
        if video_processor:
            video_processor.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# def get_improved_camera_calibration():
#     """Улучшенная калибровка с учетом дисторсий"""
#     camera_matrix = np.array([[920, 0, 640],
#                               [0, 920, 360], 
#                               [0, 0, 1]], dtype=np.float32)
    
#     # Добавляем реалистичные коэффициенты дисторсии
#     dist_coeffs = np.array([-0.2, 0.1, 0.001, 0.001, 0.0], dtype=np.float32)
    
#     return camera_matrix, dist_coeffs

# def main():
#     """Главная функция приложения"""
    
#     # --- Конфигурация ---
#     ARUCO_DICTS = [
#         cv2.aruco.DICT_4X4_50,
#         cv2.aruco.DICT_4X4_100,
#         cv2.aruco.DICT_5X5_50,
#     ]
    
#     # Псевдокалибровка камеры
#     # camera_matrix = np.array([[920, 0, 640],
#     #                           [0, 920, 360],
#     #                           [0, 0, 1]], dtype=np.float32)
#     # dist_coeffs = np.zeros((5, 1))

#     camera_matrix, dist_coeffs = get_improved_camera_calibration()
    
#     # --- Обработчик завершения ---
#     def handle_exit(signum=None, frame=None):
#         print("\n🛑 Завершение работы...")
#         cv2.destroyAllWindows()
#         sys.exit(0)
    
#     signal.signal(signal.SIGINT, handle_exit)
    
#     try:
#         # Инициализация компонентов
#         video_processor = VideoProcessor("../../assets/ar_test_video.MOV")
#         aruco_detector = AdvancedArUcoDetector(ARUCO_DICTS, camera_matrix, dist_coeffs)
        
#         # --- Главный цикл обработки ---
#         while True:
#             ret, frame = video_processor.read_frame()
#             if not ret:
#                 print("⚠️ Конец видео или ошибка чтения.")
#                 break
            
#             # Предобработка
#             processed_frame = aruco_detector.preProcess(frame)
            
#             # Основная обработка
#             detected = aruco_detector.process_advanced(frame)  # Используем оригинальный frame для детекции
            
#             # Постобработка
#             aruco_detector.postProcess(frame)
            
#             # Отображение результата
#             cv2.imshow("ArUco Pose Estimation", frame)
            
#             # Проверка условий выхода
#             key = cv2.waitKey(10) & 0xFF
#             if key == 27 or cv2.getWindowProperty("ArUco Pose Estimation", cv2.WND_PROP_VISIBLE) < 1:
#                 handle_exit()
                
#     except Exception as e:
#         print(f"❌ Ошибка: {e}")
#         handle_exit()


# if __name__ == "__main__":
#     main()