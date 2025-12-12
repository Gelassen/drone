# detective_detector.py
import cv2
import numpy as np
from typing import List, Dict, Any

class ArUcoDetective:
    def __init__(self):
        # ВСЕ возможные словари ArUco
        self.all_dicts = [
            cv2.aruco.DICT_4X4_50, cv2.aruco.DICT_4X4_100, cv2.aruco.DICT_4X4_250, cv2.aruco.DICT_4X4_1000,
            cv2.aruco.DICT_5X5_50, cv2.aruco.DICT_5X5_100, cv2.aruco.DICT_5X5_250, cv2.aruco.DICT_5X5_1000,
            cv2.aruco.DICT_6X6_50, cv2.aruco.DICT_6X6_100, cv2.aruco.DICT_6X6_250, cv2.aruco.DICT_6X6_1000,
            cv2.aruco.DICT_7X7_50, cv2.aruco.DICT_7X7_100, cv2.aruco.DICT_7X7_250, cv2.aruco.DICT_7X7_1000,
        ]
        
        self.parameters = cv2.aruco.DetectorParameters()
        self._setup_ultra_permissive_parameters()
    
    def _setup_ultra_permissive_parameters(self):
        """Сверх-разрешающие параметры"""
        self.parameters.adaptiveThreshWinSizeMin = 3
        self.parameters.adaptiveThreshWinSizeMax = 23
        self.parameters.adaptiveThreshWinSizeStep = 10
        self.parameters.adaptiveThreshConstant = 5
        self.parameters.minMarkerPerimeterRate = 0.01
        self.parameters.maxMarkerPerimeterRate = 8.0
        self.parameters.polygonalApproxAccuracyRate = 0.2
        self.parameters.minCornerDistanceRate = 0.01
        self.parameters.minDistanceToBorder = 0
        self.parameters.markerBorderBits = 1
        self.parameters.minOtsuStdDev = 3.0
        self.parameters.perspectiveRemovePixelPerCell = 4
        self.parameters.perspectiveRemoveIgnoredMarginPerCell = 0.25
        self.parameters.maxErroneousBitsInBorderRate = 0.8
        self.parameters.errorCorrectionRate = 0.3
    
    def find_correct_dictionary(self, frame: np.ndarray) -> Dict[str, Any]:
        """Поиск правильного словаря и ID маркеров"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = {}
        
        print("🔍 Поиск правильного словаря ArUco...")
        print("=" * 60)
        
        for dict_type in self.all_dicts:
            try:
                aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
                dict_name = self._get_dict_name(dict_type)
                
                # Пробуем 4 ориентации маркера
                for orientation in [0, 1, 2, 3]:
                    if orientation > 0:
                        # Поворачиваем параметры для тестирования разных ориентаций
                        temp_params = cv2.aruco.DetectorParameters()
                        temp_params = self.parameters
                        # Некоторые версии OpenCV имеют параметр ориентации
                    
                    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=self.parameters)
                    
                    if ids is not None and len(ids) > 0:
                        found_ids = ids.flatten().tolist()
                        results[dict_name] = {
                            'ids': found_ids,
                            'count': len(ids),
                            'corners': corners
                        }
                        
                        print(f"✅ {dict_name}: Найдено {len(ids)} маркеров, ID: {found_ids}")
                        
                        # Если нашли 4 маркера - это вероятно наш случай
                        if len(ids) >= 4:
                            print(f"🎯 ВОЗМОЖНО НАЙДЕН ПРАВИЛЬНЫЙ СЛОВАРЬ: {dict_name}")
                            return results[dict_name]
            
            except Exception as e:
                continue
        
        # Возвращаем лучший результат (с максимальным количеством маркеров)
        if results:
            best_result = max(results.items(), key=lambda x: x[1]['count'])
            print(f"🏆 ЛУЧШИЙ РЕЗУЛЬТАТ: {best_result[0]} - {best_result[1]['count']} маркеров")
            return best_result[1]
        else:
            print("❌ Не удалось найти подходящий словарь")
            return None
    
    def test_marker_orientation(self, frame: np.ndarray, dict_type: int):
        """Тестирование разных ориентаций маркеров"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
        
        print(f"\n🧭 Тестирование ориентаций для {self._get_dict_name(dict_type)}")
        
        # Создаем копии кадра с разными поворотами
        rotations = [0, 90, 180, 270]
        
        for angle in rotations:
            if angle == 0:
                rotated_frame = gray
            else:
                # Поворачиваем кадр
                h, w = gray.shape
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated_frame = cv2.warpAffine(gray, M, (w, h))
            
            corners, ids, rejected = cv2.aruco.detectMarkers(rotated_frame, aruco_dict, parameters=self.parameters)
            
            if ids is not None:
                print(f"   Поворот {angle}°: Найдено {len(ids)} маркеров")
    
    def analyze_marker_sizes(self, frame: np.ndarray):
        """Анализ размеров маркеров"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Простая детекция квадратов
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        squares = []
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            
            if len(approx) == 4:  # Квадрат/прямоугольник
                area = cv2.contourArea(contour)
                if area > 100:  # Отфильтровываем слишком маленькие
                    squares.append({
                        'area': area,
                        'perimeter': perimeter,
                        'points': approx
                    })
        
        print(f"\n📐 Найдено {len(squares)} квадратных объектов:")
        for i, square in enumerate(squares):
            print(f"   Квадрат {i+1}: площадь={square['area']:.0f}, периметр={square['perimeter']:.0f}")
    
    def _get_dict_name(self, dict_type: int) -> str:
        """Получить читаемое имя словаря"""
        dict_names = {
            cv2.aruco.DICT_4X4_50: "4X4_50", cv2.aruco.DICT_4X4_100: "4X4_100",
            cv2.aruco.DICT_4X4_250: "4X4_250", cv2.aruco.DICT_4X4_1000: "4X4_1000",
            cv2.aruco.DICT_5X5_50: "5X5_50", cv2.aruco.DICT_5X5_100: "5X5_100", 
            cv2.aruco.DICT_5X5_250: "5X5_250", cv2.aruco.DICT_5X5_1000: "5X5_1000",
            cv2.aruco.DICT_6X6_50: "6X6_50", cv2.aruco.DICT_6X6_100: "6X6_100",
            cv2.aruco.DICT_6X6_250: "6X6_250", cv2.aruco.DICT_6X6_1000: "6X6_1000",
            cv2.aruco.DICT_7X7_50: "7X7_50", cv2.aruco.DICT_7X7_100: "7X7_100",
            cv2.aruco.DICT_7X7_250: "7X7_250", cv2.aruco.DICT_7X7_1000: "7X7_1000",
        }
        return dict_names.get(dict_type, f"UNKNOWN_{dict_type}")

def main():
    cap = cv2.VideoCapture("../../assets/ar_test_video.MOV")
    detective = ArUcoDetective()
    
    print("🕵️ ДЕТЕКТИВ ARUCO - поиск правильных параметров")
    print("=" * 60)
    
    # Берем первый кадр для анализа
    ret, frame = cap.read()
    if not ret:
        print("Не удалось прочитать видео")
        return
    
    # 1. Анализ размеров маркеров
    detective.analyze_marker_sizes(frame)
    
    # 2. Поиск правильного словаря
    result = detective.find_correct_dictionary(frame)
    
    # 3. Если нашли что-то, тестируем ориентацию
    if result:
        # Найдем соответствующий dict_type
        for dict_type in detective.all_dicts:
            if detective._get_dict_name(dict_type) in str(result):
                detective.test_marker_orientation(frame, dict_type)
                break
    
    # 4. Визуализация результатов
    if result and 'corners' in result:
        debug_frame = frame.copy()
        cv2.aruco.drawDetectedMarkers(debug_frame, result['corners'], np.array(result['ids']))
        
        # Добавляем информацию
        cv2.putText(debug_frame, f"Dictionary: {list(result.keys())[0]}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(debug_frame, f"Markers found: {result['count']}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Detective Results", debug_frame)
        cv2.waitKey(0)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()