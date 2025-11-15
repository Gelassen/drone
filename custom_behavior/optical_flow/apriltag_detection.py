import cv2
import numpy as np
import apriltag
import time
import os

class StableAprilTagDetector:
    def __init__(self, camera_matrix=None, dist_coeffs=None, family='tag16h5'):
        """
        Инициализация стабильного детектора AprilTags
        """
        # Создаем options с правильными параметрами
        options = apriltag.DetectorOptions()
        options.families = family
        options.nthreads = 4
        options.quad_decimate = 1.0
        options.quad_blur = 0.0  # Заменяем quad_sigma
        options.refine_edges = True
        options.decode_sharpening = 0.25
        options.debug = False
        
        self.detector = apriltag.Detector(options)
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.axis_length = 0.1
        
        # Для стабилизации
        self.last_detections = {}
        self.stability_threshold = 3  # Количество кадров для стабилизации
        self.tracking_enabled = True
        
        print(f"Инициализирован детектор для {family} с настройками стабильности")

    def preprocess_image(self, image):
        """
        Предобработка изображения для улучшения детектирования
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Увеличиваем контраст
        gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)
        
        # Легкое размытие для уменьшения шума
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        return gray

    def stabilize_detections(self, current_detections):
        """
        Стабилизация детектирования между кадрами
        """
        stabilized = []
        current_time = time.time()
        
        # Обновляем предыдущие детекции
        for detection in current_detections:
            tag_id = detection.tag_id
            center = detection.corners.mean(axis=0)
            
            if tag_id in self.last_detections:
                # Проверяем расстояние от предыдущей позиции
                prev_center = self.last_detections[tag_id]['center']
                distance = np.linalg.norm(center - prev_center)
                
                # Если маркер не переместился слишком далеко, считаем его стабильным
                if distance < 50:  # пикселей
                    self.last_detections[tag_id]['count'] += 1
                    self.last_detections[tag_id]['center'] = center
                    self.last_detections[tag_id]['last_seen'] = current_time
                    
                    # Если маркер стабилен в нескольких кадрах, добавляем его
                    if self.last_detections[tag_id]['count'] >= self.stability_threshold:
                        stabilized.append(detection)
                else:
                    # Сброс счетчика для переместившегося маркера
                    self.last_detections[tag_id] = {
                        'count': 1,
                        'center': center,
                        'last_seen': current_time
                    }
            else:
                # Новый маркер
                self.last_detections[tag_id] = {
                    'count': 1,
                    'center': center,
                    'last_seen': current_time
                }
        
        # Удаляем старые детекции (не видели больше 1 секунды)
        expired_ids = []
        for tag_id, data in self.last_detections.items():
            if current_time - data['last_seen'] > 1.0:
                expired_ids.append(tag_id)
        
        for tag_id in expired_ids:
            del self.last_detections[tag_id]
        
        return stabilized

    def detect_apriltags_stable(self, image):
        """
        Стабильное детектирование AprilTags
        """
        # Предобработка
        processed = self.preprocess_image(image)
        
        # Детектирование
        try:
            raw_detections = self.detector.detect(processed)
            
            if self.tracking_enabled:
                stabilized_detections = self.stabilize_detections(raw_detections)
                return stabilized_detections
            else:
                return raw_detections
                
        except Exception as e:
            print(f"Ошибка детектирования: {e}")
            return []

    def estimate_pose(self, corners, tag_size=0.1):
        """Оценка позы маркера"""
        if self.camera_matrix is None:
            return None, None
            
        obj_points = np.array([
            [-tag_size/2, -tag_size/2, 0],
            [ tag_size/2, -tag_size/2, 0],
            [ tag_size/2,  tag_size/2, 0],
            [-tag_size/2,  tag_size/2, 0]
        ], dtype=np.float32)
        
        success, rvec, tvec = cv2.solvePnP(
            obj_points, 
            corners.astype(np.float32), 
            self.camera_matrix, 
            self.dist_coeffs
        )
        
        return (rvec, tvec) if success else (None, None)

    def draw_detection_info(self, image, detection, stability_count=1):
        """Отрисовка информации о детектированном маркере"""
        corners = detection.corners.astype(int)
        tag_id = detection.tag_id
        
        # Цвет в зависимости от стабильности
        if stability_count >= self.stability_threshold:
            color = (0, 255, 0)  # Зеленый - стабильный
            thickness = 3
        else:
            color = (0, 165, 255)  # Оранжевый - нестабильный
            thickness = 2
        
        # Рисуем bounding box
        for i in range(4):
            cv2.line(image, 
                    tuple(corners[i]), 
                    tuple(corners[(i + 1) % 4]), 
                    color, thickness)
        
        # Центр
        center = corners.mean(axis=0).astype(int)
        cv2.circle(image, tuple(center), 6, (255, 0, 255), -1)
        
        # Текст с информацией
        stability_text = "STABLE" if stability_count >= self.stability_threshold else f"UNSTABLE ({stability_count}/{self.stability_threshold})"
        text = f"ID: {tag_id} - {stability_text}"
        cv2.putText(image, text, 
                   (center[0] - 60, center[1] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Оценка позы и отображение осей
        if self.camera_matrix is not None:
            rvec, tvec = self.estimate_pose(detection.corners)
            if rvec is not None and tvec is not None:
                self.draw_axes(image, rvec, tvec)

    def draw_axes(self, image, rvec, tvec):
        """Отрисовка осей координат"""
        axis_points = np.float32([
            [0, 0, 0],
            [self.axis_length, 0, 0],
            [0, self.axis_length, 0],
            [0, 0, self.axis_length]
        ]).reshape(-1, 3)
        
        img_points, _ = cv2.projectPoints(
            axis_points, rvec, tvec, 
            self.camera_matrix, self.dist_coeffs
        )
        
        img_points = img_points.reshape(-1, 2).astype(int)
        origin = tuple(img_points[0])
        
        # Оси
        cv2.arrowedLine(image, origin, tuple(img_points[1]), (0, 0, 255), 2)
        cv2.arrowedLine(image, origin, tuple(img_points[2]), (0, 255, 0), 2)
        cv2.arrowedLine(image, origin, tuple(img_points[3]), (255, 0, 0), 2)

def process_video_file(video_path, output_path=None):
    """
    Обработка видеофайла с AprilTags
    """
    # Проверяем существование файла
    if not os.path.exists(video_path):
        print(f"Ошибка: Файл {video_path} не найден")
        return
    
    # Матрица камеры (может потребоваться калибровка для вашей камеры)
    camera_matrix = np.array([
        [800, 0, 640],   # fx, 0, cx (для видео 1280x720)
        [0, 800, 360],   # 0, fy, cy
        [0, 0, 1]
    ], dtype=np.float32)
    
    dist_coeffs = np.zeros((4, 1))
    
    # Инициализация детектора
    detector = StableAprilTagDetector(camera_matrix, dist_coeffs, family='tag16h5')
    
    # Открываем видео файл
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видео файл {video_path}")
        return
    
    # Получаем информацию о видео
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"=== Обработка видеофайла ===")
    print(f"Файл: {video_path}")
    print(f"Разрешение: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Всего кадров: {total_frames}")
    print("Q - Выход")
    print("P - Пауза/Продолжить")
    print("S - Сохранить текущий кадр")
    print("==========================")
    
    # Настройка для сохранения видео (если нужно)
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Достигнут конец видео")
                break
            
            frame_count += 1
            
            # Детектирование
            results = detector.detect_apriltags_stable(frame)
            
            # Отрисовка результатов
            for detection in results:
                tag_id = detection.tag_id
                stability_count = detector.last_detections.get(tag_id, {}).get('count', 1)
                detector.draw_detection_info(frame, detection, stability_count)
            
            # Информация на кадре
            cv2.putText(frame, f"Кадр: {frame_count}/{total_frames}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Маркеров: {len(results)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Отображение ID найденных маркеров
            if results:
                ids_text = "ID: " + ", ".join([str(det.tag_id) for det in results])
                cv2.putText(frame, ids_text, 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Сохранение кадра (если нужно)
            if output_path:
                out.write(frame)
        
        # Отображение
        cv2.imshow('AprilTag Detection - Video File', frame)
        
        # Обработка клавиш
        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print(f"Видео {'приостановлено' if paused else 'продолжено'}")
        elif key == ord('s'):
            filename = f"frame_{frame_count:04d}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Кадр сохранен: {filename}")
    
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()
    
    print(f"Обработка завершена. Обработано кадров: {frame_count}")

def process_webcam():
    """
    Обработка видео с веб-камеры
    """
    camera_matrix = np.array([
        [800, 0, 320],
        [0, 800, 240], 
        [0, 0, 1]
    ], dtype=np.float32)
    
    dist_coeffs = np.zeros((4, 1))
    
    detector = StableAprilTagDetector(camera_matrix, dist_coeffs, family='tag16h5')
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("=== AprilTag Детектор (Веб-камера) ===")
    print("Q - Выход")
    print("S - Сохранить кадр") 
    print("T - Вкл/Выкл трекинг стабильности")
    print("1-9 - Настройка стабильности")
    print("==========================")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Детектирование
        results = detector.detect_apriltags_stable(frame)
        
        # Отрисовка результатов
        for detection in results:
            tag_id = detection.tag_id
            stability_count = detector.last_detections.get(tag_id, {}).get('count', 1)
            detector.draw_detection_info(frame, detection, stability_count)
        
        # FPS
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - start_time)
            start_time = time.time()
            cv2.putText(frame, f"FPS: {fps:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Информация
        cv2.putText(frame, f"Маркеров: {len(results)}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        if results:
            stable_ids = [str(det.tag_id) for det in results 
                         if detector.last_detections.get(det.tag_id, {}).get('count', 0) >= detector.stability_threshold]
            if stable_ids:
                cv2.putText(frame, f"Стабильные: {', '.join(stable_ids)}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow('AprilTag Detector - Webcam', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"webcam_frame_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Сохранено: {filename}")
        elif key == ord('t'):
            detector.tracking_enabled = not detector.tracking_enabled
            print(f"Трекинг: {'ВКЛ' if detector.tracking_enabled else 'ВЫКЛ'}")
        elif ord('1') <= key <= ord('9'):
            detector.stability_threshold = key - ord('0')
            print(f"Порог стабильности: {detector.stability_threshold} кадров")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Обработка видео файла
    video_path = "../../assets/ar_test_video.MOV"
    process_video_file(video_path)
    
    # Или обработка с веб-камеры (раскомментируйте следующую строку)
    # process_webcam()