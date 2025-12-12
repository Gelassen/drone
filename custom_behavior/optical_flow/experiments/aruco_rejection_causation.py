# rejection_analyzer.py
import cv2
import numpy as np
from typing import List, Tuple

class RejectionAnalyzer:
    def __init__(self, aruco_dicts: List[int]):
        self.aruco_dicts = aruco_dicts
        
        try:
            self.parameters = cv2.aruco.DetectorParameters()
        except AttributeError:
            self.parameters = cv2.aruco.DetectorParameters_create()
        
        # СНИЖАЕМ ТРЕБОВАНИЯ для принятия маркеров
        self.parameters.minMarkerPerimeterRate = 0.008    # Очень маленькие маркеры
        self.parameters.maxMarkerPerimeterRate = 8.0      # Очень большие маркеры
        self.parameters.polygonalApproxAccuracyRate = 0.1 # Менее строгая аппроксимация
        self.parameters.minCornerDistanceRate = 0.02      # Близкие углы разрешены
        self.parameters.minDistanceToBorder = 0           # Маркеры у границы
        self.parameters.markerBorderBits = 1              # Уже граница
        self.parameters.minOtsuStdDev = 4.0               # Ниже порог для темных
        self.parameters.perspectiveRemovePixelPerCell = 6 # Выше разрешение
        self.parameters.perspectiveRemoveIgnoredMarginPerCell = 0.15
        self.parameters.maxErroneousBitsInBorderRate = 0.5 # Больше шума в границе
        self.parameters.errorCorrectionRate = 0.6         # Меньше коррекции ошибок

    def analyze_rejection_reasons(self, frame: np.ndarray):
        """Анализ причин отвержения маркеров"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        debug_frame = frame.copy()
        height, width = frame.shape[:2]
        
        print(f"\n=== АНАЛИЗ ПРИЧИН ОТВЕРЖЕНИЯ ===")
        
        for dict_type in self.aruco_dicts:
            aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
            dict_name = self._get_dict_name(dict_type)
            
            print(f"\n--- {dict_name} ---")
            
            # Детекция с текущими параметрами
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=self.parameters)
            
            if ids is not None:
                print(f"✅ Найдено маркеров: {len(ids)}")
                cv2.aruco.drawDetectedMarkers(debug_frame, corners, ids, borderColor=(0, 255, 0))
            
            # Анализ каждого отвергнутого кандидата
            if rejected is not None:
                print(f"❌ Отвергнуто кандидатов: {len(rejected)}")
                
                for i, candidate in enumerate(rejected):
                    candidate_analysis = self._analyze_single_candidate(gray, candidate, aruco_dict)
                    
                    # Визуализация с цветом по причине отвержения
                    color = self._get_rejection_color(candidate_analysis['reason'])
                    points = candidate.reshape(-1, 2).astype(int)
                    cv2.polylines(debug_frame, [points], True, color, 2)
                    
                    # Подпись с причиной
                    center = np.mean(points, axis=0).astype(int)
                    cv2.putText(debug_frame, f"{candidate_analysis['reason']}", 
                               (center[0]-50, center[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    
                    print(f"  Кандидат {i+1}: {candidate_analysis['reason']}")
                    print(f"    Размер: {candidate_analysis['size']}, "
                          f"Соотношение: {candidate_analysis['aspect_ratio']:.2f}, "
                          f"Контраст: {candidate_analysis['contrast']:.1f}")
        
        return debug_frame
    
    def _analyze_single_candidate(self, gray: np.ndarray, candidate: np.ndarray, aruco_dict) -> dict:
        """Анализ одного отвергнутого кандидата"""
        points = candidate.reshape(-1, 2)
        
        # Геометрический анализ
        perimeter = cv2.arcLength(points, True)
        area = cv2.contourArea(points)
        
        # Размер и форма
        rect = cv2.minAreaRect(points)
        width, height = rect[1]
        aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 0
        
        # Анализ контраста внутри кандидата
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.fillConvexPoly(mask, points.astype(int), 255)
        mean_intensity = cv2.mean(gray, mask=mask)[0]
        
        # Причина отвержения (упрощенный анализ)
        reason = "unknown"
        
        if perimeter < 40:  # Слишком маленький
            reason = "too_small"
        elif aspect_ratio > 3:  # Слишком вытянутый
            reason = "bad_shape"  
        elif area < 100:  # Слишком маленькая площадь
            reason = "small_area"
        elif mean_intensity < 50 or mean_intensity > 200:  # Проблемы с контрастом
            reason = "bad_contrast"
        else:
            reason = "pattern_rejection"  # Не прошел проверку паттерна
        
        return {
            'reason': reason,
            'size': perimeter,
            'aspect_ratio': aspect_ratio,
            'contrast': mean_intensity
        }
    
    def _get_rejection_color(self, reason: str) -> Tuple[int, int, int]:
        """Цвет для визуализации по причине отвержения"""
        colors = {
            'too_small': (255, 0, 255),      # Фиолетовый - слишком маленький
            'bad_shape': (255, 255, 0),      # Голубой - плохая форма
            'small_area': (0, 255, 255),     # Желтый - маленькая площадь  
            'bad_contrast': (0, 165, 255),   # Оранжевый - проблемы с контрастом
            'pattern_rejection': (0, 0, 255), # Красный - не прошел проверку паттерна
            'unknown': (128, 128, 128)       # Серый - неизвестно
        }
        return colors.get(reason, (128, 128, 128))
    
    def _get_dict_name(self, dict_type: int) -> str:
        names = {
            cv2.aruco.DICT_4X4_50: "DICT_4X4_50",
            cv2.aruco.DICT_4X4_100: "DICT_4X4_100", 
            cv2.aruco.DICT_5X5_50: "DICT_5X5_50"
        }
        return names.get(dict_type, f"UNKNOWN_{dict_type}")

# Основная функция с тестированием разных параметров
def main():
    ARUCO_DICTS = [cv2.aruco.DICT_4X4_50, cv2.aruco.DICT_4X4_100, cv2.aruco.DICT_5X5_50]
    
    cap = cv2.VideoCapture("../../assets/ar_test_video.MOV")
    analyzer = RejectionAnalyzer(ARUCO_DICTS)
    
    print("=== АНАЛИЗ ПРИЧИН ОТВЕРЖЕНИЯ МАРКЕРОВ ===")
    print("Цвета отвержения:")
    print("🟣 Фиолетовый - Слишком маленький")
    print("🔵 Голубой - Плохая форма") 
    print("🟡 Желтый - Маленькая площадь")
    print("🟠 Оранжевый - Проблемы с контрастом")
    print("🔴 Красный - Не прошел проверку паттерна")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        debug_frame = analyzer.analyze_rejection_reasons(frame)
        cv2.imshow("Rejection Analysis", debug_frame)
        
        key = cv2.waitKey(100) & 0xFF
        if key == 27:
            break
        elif key == ord(' '):  # Пауза на пробел
            cv2.waitKey(0)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()