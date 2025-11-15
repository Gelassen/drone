import cv2
import numpy as np

try:
    import apriltag
    
    # Проверка доступных атрибутов
    print("Доступные атрибуты apriltag:", [x for x in dir(apriltag) if not x.startswith('_')])
    
    # Создание детектора с проверкой доступных параметров
    if hasattr(apriltag, 'DetectorOptions'):
        options = apriltag.DetectorOptions(families='tag36h11')
        detector = apriltag.Detector(options)
    else:
        detector = apriltag.Detector()
        
except Exception as e:
    print(f"Ошибка импорта apriltag: {e}")
    print("Попробуйте установить другую версию:")
    print("pip uninstall apriltag")
    print("pip install pupil-apriltags")

def minimal_apriltag_detection():
    """Минимальная рабочая версия"""
    try:
        import apriltag
        
        # Простой способ инициализации
        detector = apriltag.Detector()
        
        cap = cv2.VideoCapture("../../assets/ar_test_video.MOV")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results = detector.detect(gray)
            
            for detection in results:
                corners = detection.corners.astype(int)
                tag_id = detection.tag_id
                
                # Рисуем bounding box
                for i in range(4):
                    cv2.line(frame, tuple(corners[i]), tuple(corners[(i+1)%4]), (0,255,0), 2)
                
                # Показываем ID
                center = corners.mean(axis=0).astype(int)
                cv2.putText(frame, f"ID: {tag_id}", tuple(center), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            
            cv2.imshow('AprilTags', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Ошибка: {e}")

# Запуск минимальной версии
minimal_apriltag_detection()