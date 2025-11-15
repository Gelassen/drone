# corrected_diagnostic.py
import cv2
import numpy as np
from typing import List, Tuple

class CorrectedDiagnosticDetector:
    def __init__(self, aruco_dicts: List[int], camera_matrix: np.ndarray, 
                 dist_coeffs: np.ndarray, marker_length: float = 0.1):
        self.aruco_dicts = aruco_dicts
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.marker_length = marker_length
        
        try:
            self.parameters = cv2.aruco.DetectorParameters()
        except AttributeError:
            self.parameters = cv2.aruco.DetectorParameters_create()
        
        # Ослабляем параметры для лучшей детекции
        self.parameters.minMarkerPerimeterRate = 0.01
        self.parameters.maxMarkerPerimeterRate = 10.0
        self.parameters.adaptiveThreshConstant = 3
        self.parameters.polygonalApproxAccuracyRate = 0.1
    
    def analyze_and_visualize(self, frame: np.ndarray) -> Tuple[dict, np.ndarray]:
        """Анализ и визуализация в одном методе"""
        height, width = frame.shape[:2]
        debug_frame = frame.copy()
        
        # Анализ яркости
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_analysis = {
            'top': np.mean(gray[0:height//2, :]),
            'bottom': np.mean(gray[height//2:, :]),
            'overall': np.mean(gray)
        }
        
        # Рисуем разделительную линию
        cv2.line(debug_frame, (0, height//2), (width, height//2), (100, 100, 100), 2)
        cv2.putText(debug_frame, f"TOP: {brightness_analysis['top']:.1f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(debug_frame, f"BOTTOM: {brightness_analysis['bottom']:.1f}", 
                   (10, height//2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        results = {}
        
        for dict_type in self.aruco_dicts:
            aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray, aruco_dict, parameters=self.parameters
            )
            
            dict_name = self._get_dict_name(dict_type)
            
            # ВИЗУАЛИЗАЦИЯ НАЙДЕННЫХ МАРКЕРОВ (зеленый)
            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(debug_frame, corners, ids)
                print(f"✅ {dict_name}: Найдены маркеры {ids.flatten().tolist()}")
                
                # Оценка позы для найденных маркеров
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, self.marker_length, self.camera_matrix, self.dist_coeffs
                )
                
                # Рисуем оси для найденных маркеров
                for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
                    self.draw_axes_safe(debug_frame, rvec, tvec)
                    
            # ВИЗУАЛИЗАЦИЯ ОТВЕРГНУТЫХ КАНДИДАТОВ (красный)
            if rejected is not None and len(rejected) > 0:
                for candidate in rejected:
                    points = candidate.reshape(-1, 2).astype(int)
                    cv2.polylines(debug_frame, [points], True, (0, 0, 255), 2)
                print(f"❌ {dict_name}: Отвергнуто кандидатов: {len(rejected)}")
            
            results[dict_name] = {
                'found_ids': ids.flatten().tolist() if ids is not None else [],
                'found_count': len(ids) if ids is not None else 0,
                'rejected_count': len(rejected) if rejected is not None else 0
            }
        
        return results, debug_frame
    
    def draw_axes_safe(self, frame, rvec, tvec, length=0.05):
        """Рисование осей для найденных маркеров"""
        try:
            if hasattr(cv2, "drawFrameAxes"):
                cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, length)
            elif hasattr(cv2.aruco, "drawAxis"):
                cv2.aruco.drawAxis(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, length)
        except Exception as e:
            print(f"Ошибка при рисовании осей: {e}")
    
    def _get_dict_name(self, dict_type: int) -> str:
        names = {
            cv2.aruco.DICT_4X4_50: "DICT_4X4_50",
            cv2.aruco.DICT_4X4_100: "DICT_4X4_100", 
            cv2.aruco.DICT_5X5_50: "DICT_5X5_50"
        }
        return names.get(dict_type, f"UNKNOWN_{dict_type}")

# Упрощенный main для тестирования
def main():
    ARUCO_DICTS = [cv2.aruco.DICT_4X4_50, cv2.aruco.DICT_4X4_100, cv2.aruco.DICT_5X5_50]
    
    camera_matrix = np.array([[920, 0, 640], [0, 920, 360], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((5, 1))
    
    cap = cv2.VideoCapture("../../assets/ar_test_video.MOV")
    detector = CorrectedDiagnosticDetector(ARUCO_DICTS, camera_matrix, dist_coeffs)
    
    print("=== РЕЖИМ ДИАГНОСТИКИ ===")
    print("✅ ЗЕЛЕНЫЕ рамки - НАЙДЕННЫЕ маркеры")
    print("❌ КРАСНЫЕ рамки - ОТВЕРГНУТЫЕ кандидаты")
    print("📊 Белый текст - яркость зон")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        results, debug_frame = detector.analyze_and_visualize(frame)
        
        cv2.imshow("Diagnostic - CORRECTED", debug_frame)
        
        key = cv2.waitKey(100) & 0xFF
        if key == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()