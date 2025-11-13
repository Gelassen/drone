import cv2
import numpy as np
import sys
import signal
from typing import List, Tuple, Optional, Dict, Any

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


class ArUcoDetector:
    """Основной класс для детекции ArUco маркеров"""
    
    def __init__(self, aruco_dicts: List[int], camera_matrix: np.ndarray, 
                 dist_coeffs: np.ndarray, marker_length: float = 0.1):
        self.aruco_dicts = aruco_dicts
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.marker_length = marker_length
        self.detected_markers: List[ArucoMarker] = []
        self.current_dict = None
        
        # Инициализация параметров детектора
        try:
            self.parameters = cv2.aruco.DetectorParameters()
            print("OpenCV >= 4.7 is used")
        except AttributeError:
            self.parameters = cv2.aruco.DetectorParameters_create()
            print("OpenCV < 4.7 (legacy) is used")
        
        self._setup_parameters()
    
    def _setup_parameters(self) -> None:
        """Настройка параметров детектора"""
        self.parameters.adaptiveThreshConstant = 5
        self.parameters.minMarkerPerimeterRate = 0.02
        self.parameters.maxMarkerPerimeterRate = 4.0
        self.parameters.polygonalApproxAccuracyRate = 0.02
    
    def preProcess(self, frame: np.ndarray) -> np.ndarray:
        """Предобработка кадра"""
        return FrameProcessor.preprocess(frame)
    
    def process(self, frame: np.ndarray) -> bool:
        """Основной процесс детекции маркеров"""
        self.detected_markers.clear()
        
        for dict_type in self.aruco_dicts:
            aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
            self.current_dict = dict_type
            
            # Детекция маркеров
            corners, ids, rejected = cv2.aruco.detectMarkers(
                frame, aruco_dict, parameters=self.parameters
            )
            
            if ids is not None and len(ids) > 0:
                self._process_detected_markers(corners, ids)
                return True
        
        return False
    
    def _process_detected_markers(self, corners: List[np.ndarray], ids: np.ndarray) -> None:
        """Обработка обнаруженных маркеров"""
        # Оценка позы маркеров
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, self.marker_length, self.camera_matrix, self.dist_coeffs
        )
        
        # Создание объектов маркеров
        for i, marker_id in enumerate(ids):
            rvec = rvecs[i] if rvecs is not None else None
            tvec = tvecs[i] if tvecs is not None else None
            
            marker = ArucoMarker(marker_id[0], corners[i], rvec, tvec)
            if rvec is not None:
                marker.calculate_euler_angles()
            
            self.detected_markers.append(marker)
    
    def postProcess(self, frame: np.ndarray) -> None:
        """Постобработка - отрисовка результатов"""
        if not self.detected_markers:
            cv2.putText(frame, "Marker not detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
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
                
                # Вывод в консоль
                print(f"[DICT_{self.current_dict}] ID:{marker.id} "
                      f"Roll: {marker.roll:7.2f}°, Pitch: {marker.pitch:7.2f}°, "
                      f"Yaw: {marker.yaw:7.2f}°")


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
