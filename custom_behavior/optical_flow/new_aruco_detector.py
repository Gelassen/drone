# aruco_detector.py
import cv2
import numpy as np
import sys
import signal
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum

class ArucoMarker:
    """Класс для представления обнаруженного маркера"""
    
    def __init__(self, marker_id: int, corners: np.ndarray, 
                 rvec: Optional[np.ndarray] = None, 
                 tvec: Optional[np.ndarray] = None):
        self.id = marker_id
        self.corners = corners
        self.rvec = rvec
        self.tvec = tvec
        self.roll = None
        self.pitch = None
        self.yaw = None
        
    def calculate_euler_angles(self) -> None:
        """Расчёт углов Эйлера из вектора вращения"""
        if self.rvec is not None:
            R, _ = cv2.Rodrigues(self.rvec)
            self.roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
            self.pitch = np.degrees(np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2)))
            self.yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    
    def get_euler_angles_str(self) -> str:
        """Получить строку с углами Эйлера"""
        if all(angle is not None for angle in [self.roll, self.pitch, self.yaw]):
            return f"R:{self.roll:6.1f} P:{self.pitch:6.1f} Y:{self.yaw:6.1f}"
        return "Angles not available"

class FrameProcessor:
    """Класс для предварительной обработки кадров"""
    
    @staticmethod
    def preprocess(frame: np.ndarray) -> np.ndarray:
        """Предобработка кадра для улучшения детекции"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        return gray
    
    @staticmethod
    def draw_axes_safe(frame: np.ndarray, camera_matrix: np.ndarray, 
                      dist_coeffs: np.ndarray, rvec: np.ndarray, 
                      tvec: np.ndarray, length: float = 0.05) -> None:
        """Безопасная отрисовка 3D-осей"""
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
        axis_points = np.float32([
            [0, 0, 0],
            [length, 0, 0],
            [0, length, 0],
            [0, 0, length]
        ])
        proj, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
        proj = proj.reshape(-1, 2).astype(int)
        cv2.line(frame, tuple(proj[0]), tuple(proj[1]), (0, 0, 255), 2)  # X (красный)
        cv2.line(frame, tuple(proj[0]), tuple(proj[2]), (0, 255, 0), 2)  # Y (зелёный)
        cv2.line(frame, tuple(proj[0]), tuple(proj[3]), (255, 0, 0), 2)  # Z (синий)



class VideoProcessor:
    """Класс для обработки видео потока"""
    
    def __init__(self, video_source: str):
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            raise ValueError(f"❌ Не удалось открыть видео: {video_source}")
        print("✅ Видео открыто. Нажмите ESC или закройте окно для выхода.")
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Чтение кадра из видео"""
        ret, frame = self.cap.read()
        return ret, frame if ret else None
    
    def release(self) -> None:
        """Освобождение ресурсов"""
        self.cap.release()


class DetectionStrategy(Enum):
    STANDARD = "standard"
    MULTI_PASS = "multi_pass"
    REGION_AWARE = "region_aware"
    ADAPTIVE = "adaptive"

class AruCoDetector:
    """Основной класс для детекции ArUco маркеров с гибкой стратегией"""
    
    def __init__(self, aruco_dicts: List[int], camera_matrix: np.ndarray, 
                 dist_coeffs: np.ndarray, marker_length: float = 0.1,
                 strategy: DetectionStrategy = DetectionStrategy.ADAPTIVE):
        self.aruco_dicts = aruco_dicts
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.marker_length = marker_length
        self.detected_markers: List[ArucoMarker] = []
        self.current_dict = None
        self.strategy = strategy
        
        # Инициализация параметров детектора
        try:
            self.parameters = cv2.aruco.DetectorParameters()
            print("OpenCV >= 4.7 is used")
        except AttributeError:
            self.parameters = cv2.aruco.DetectorParameters_create()
            print("OpenCV < 4.7 (legacy) is used")
        
        self._setup_standard_parameters()
        self._setup_aggressive_parameters()
        self._setup_conservative_parameters()
    
    def _setup_standard_parameters(self) -> None:
        """Стандартные параметры"""
        self.standard_params = cv2.aruco.DetectorParameters()
        self.standard_params.adaptiveThreshConstant = 5
        self.standard_params.minMarkerPerimeterRate = 0.02
        self.standard_params.maxMarkerPerimeterRate = 4.0
        self.standard_params.polygonalApproxAccuracyRate = 0.02
    
    def _setup_aggressive_parameters(self) -> None:
        """Агрессивные параметры для сложных случаев"""
        self.aggressive_params = cv2.aruco.DetectorParameters()
        self.aggressive_params.minMarkerPerimeterRate = 0.008  # Очень маленькие маркеры
        self.aggressive_params.maxMarkerPerimeterRate = 8.0    # Очень большие маркеры
        self.aggressive_params.adaptiveThreshConstant = 3
        self.aggressive_params.minCornerDistanceRate = 0.02
        self.aggressive_params.polygonalApproxAccuracyRate = 0.05
        self.aggressive_params.maxErroneousBitsInBorderRate = 0.8
        self.aggressive_params.errorCorrectionRate = 0.6
    
    def _setup_conservative_parameters(self) -> None:
        """Консервативные параметры для надежности"""
        self.conservative_params = cv2.aruco.DetectorParameters()
        self.conservative_params.minMarkerPerimeterRate = 0.03
        self.conservative_params.adaptiveThreshConstant = 7
        self.conservative_params.minCornerDistanceRate = 0.05
    
    def preProcess(self, frame: np.ndarray) -> np.ndarray:
        """Умная предобработка с анализом гистограммы"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Анализ гистограммы для определения оптимальной обработки
        mean_brightness = np.mean(gray)
        
        if mean_brightness < 80:  # Темное изображение
            # Агрессивное улучшение контраста
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            processed = clahe.apply(gray)
        elif mean_brightness > 180:  # Слишком яркое
            # Снижение контраста
            processed = cv2.normalize(gray, None, 100, 200, cv2.NORM_MINMAX)
        else:  # Нормальная яркость
            processed = cv2.equalizeHist(gray)
        
        # Легкое размытие для уменьшения шума
        processed = cv2.GaussianBlur(processed, (3, 3), 0)
        return processed
    
    def process(self, frame: np.ndarray) -> bool:
        """Универсальный метод process с выбором стратегии"""
        self.detected_markers.clear()
        
        if self.strategy == DetectionStrategy.STANDARD:
            return self._process_standard(frame)
        elif self.strategy == DetectionStrategy.MULTI_PASS:
            return self._process_multi_pass(frame)
        elif self.strategy == DetectionStrategy.REGION_AWARE:
            return self._process_region_aware(frame)
        elif self.strategy == DetectionStrategy.ADAPTIVE:
            return self._process_adaptive(frame)
        else:
            return self._process_standard(frame)
    
    def _process_standard(self, frame: np.ndarray) -> bool:
        """Стандартная детекция"""
        processed_frame = self.preProcess(frame)
        return self._detect_with_params(processed_frame, self.standard_params)
    
    def _process_multi_pass(self, frame: np.ndarray) -> bool:
        """Многопроходная детекция с разными параметрами"""
        processed_frame = self.preProcess(frame)
        
        # Попытка 1: Стандартные параметры
        if self._detect_with_params(processed_frame, self.standard_params):
            return True
        
        # Попытка 2: Агрессивные параметры
        if self._detect_with_params(processed_frame, self.aggressive_params):
            return True
            
        # Попытка 3: Оригинальный кадр с агрессивными параметрами
        if self._detect_with_params(frame, self.aggressive_params):
            return True
            
        return False
    
    def _process_region_aware(self, frame: np.ndarray) -> bool:
        """Детекция с разделением на регионы"""
        height, width = frame.shape[:2]
        
        # Верхняя половина
        top_region = frame[0:height//2, :]
        top_processed = self.preProcess(top_region)
        if self._detect_with_params(top_processed, self.standard_params, y_offset=0):
            return True
        
        # Нижняя половина (более агрессивная обработка)
        bottom_region = frame[height//2:, :]
        bottom_gray = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)
        
        # Особо агрессивная обработка для нижней части
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        bottom_processed = clahe.apply(bottom_gray)
        bottom_processed = cv2.medianBlur(bottom_processed, 3)
        
        if self._detect_with_params(bottom_processed, self.aggressive_params, y_offset=height//2):
            return True
            
        return False
    
    def _process_adaptive(self, frame: np.ndarray) -> bool:
        """Адаптивная стратегия на основе анализа кадра"""
        # Анализ характеристик кадра
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # Выбор стратегии на основе анализа
        if contrast < 25:  # Низкий контраст
            # Используем агрессивные методы
            return self._process_region_aware(frame)
        elif mean_brightness < 70 or mean_brightness > 200:  # Проблемы с освещением
            return self._process_multi_pass(frame)
        else:
            return self._process_standard(frame)
    
    def _detect_with_params(self, frame: np.ndarray, params, y_offset: int = 0) -> bool:
        """Вспомогательный метод для детекции с заданными параметрами"""
        for dict_type in self.aruco_dicts:
            aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
            
            # Для grayscale изображений
            if len(frame.shape) == 2:
                detection_frame = frame
            else:
                detection_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            corners, ids, rejected = cv2.aruco.detectMarkers(
                detection_frame, aruco_dict, parameters=params
            )
            
            if ids is not None and len(ids) > 0:
                self.current_dict = dict_type
                
                # Корректируем координаты если нужно
                if y_offset > 0:
                    adjusted_corners = []
                    for corner in corners:
                        adjusted_corner = corner.copy()
                        adjusted_corner[:, :, 1] += y_offset
                        adjusted_corners.append(adjusted_corner)
                    corners = adjusted_corners
                
                self._process_detected_markers(corners, ids)
                return True
        return False
    
    def _process_detected_markers(self, corners: List[np.ndarray], ids: np.ndarray) -> None:
        """Обработка обнаруженных маркеров"""
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, self.marker_length, self.camera_matrix, self.dist_coeffs
        )
        
        for i, marker_id in enumerate(ids):
            rvec = rvecs[i] if rvecs is not None else None
            tvec = tvecs[i] if tvecs is not None else None
            
            marker = ArucoMarker(marker_id[0], corners[i], rvec, tvec)
            if rvec is not None:
                marker.calculate_euler_angles()
            
            self.detected_markers.append(marker)
    
    def postProcess(self, frame: np.ndarray) -> None:
        """Постобработка с отладочной информацией"""
        if not self.detected_markers:
            cv2.putText(frame, "Marker not detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Рисуем зоны для отладки
            self._draw_debug_zones(frame)
            return
        
        # Подготовка данных для отрисовки
        all_corners = []
        all_ids = []
        
        for marker in self.detected_markers:
            all_corners.append(marker.corners)
            all_ids.append(marker.id)
        
        # Отрисовка всех маркеров
        if all_corners:
            all_ids_array = np.array(all_ids)
            cv2.aruco.drawDetectedMarkers(frame, all_corners, all_ids_array)
        
        # Отрисовка осей и информации для каждого маркера
        for i, marker in enumerate(self.detected_markers):
            if marker.rvec is not None and marker.tvec is not None:
                FrameProcessor.draw_axes_safe(
                    frame, self.camera_matrix, self.dist_coeffs, 
                    marker.rvec, marker.tvec
                )
                
                # Отображение углов Эйлера для каждого маркера
                y_offset = 30 + i * 25
                cv2.putText(frame, f"ID:{marker.id} {marker.get_euler_angles_str()}",
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                print(f"[DICT_{self.current_dict}] ID:{marker.id} "
                      f"Roll: {marker.roll:7.2f}°, Pitch: {marker.pitch:7.2f}°, "
                      f"Yaw: {marker.yaw:7.2f}°")
        
        # Отладочная информация
        self._draw_debug_zones(frame)
        cv2.putText(frame, f"Strategy: {self.strategy.value}", 
                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _draw_debug_zones(self, frame: np.ndarray) -> None:
        """Отрисовка зон для отладки"""
        height, width = frame.shape[:2]
        cv2.line(frame, (0, height//2), (width, height//2), (255, 0, 0), 1)
        cv2.putText(frame, "TOP", (10, height//2 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(frame, "BOTTOM", (10, height//2 + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)