import cv2
import numpy as np
import apriltag

# --- AprilTag детектор ---
apriltag_detector = apriltag.Detector(apriltag.DetectorOptions(families="tag36h11"))

# --- Камера ---
video_path = "../../assets/ar_test_video.MOV"
cap = cv2.VideoCapture(video_path)

# Параметры фильтрации квадратов
MIN_AREA = 2000      # минимальная площадь квадрата (маркер)
MAX_AREA = 15000     # максимальная площадь квадрата (игнорировать лист)
ASPECT_RATIO_TOL = 0.2  # допускаемое отклонение сторон от 1:1

def find_squares(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    squares = []
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            x, y, w, h = cv2.boundingRect(approx)
            area = w * h
            aspect_ratio = w / h
            if MIN_AREA <= area <= MAX_AREA and abs(aspect_ratio - 1) < ASPECT_RATIO_TOL:
                squares.append((x, y, w, h))
    return squares

while True:
    ret, frame = cap.read()
    if not ret:
        break

    squares = find_squares(frame)
    
    for (x, y, w, h) in squares:
        # Нарисуем квадрат (для визуализации)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,255), 2)
        
        # --- Вырезаем ROI и ищем AprilTags ---
        roi = frame[y:y+h, x:x+w]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        detections = apriltag_detector.detect(gray_roi)
        
        for det in detections:
            corners = det.corners.astype(int) + np.array([x, y])
            cv2.polylines(frame, [corners], isClosed=True, color=(0,255,0), thickness=2)
            cv2.putText(frame, f"ID: {det.tag_id}", tuple(corners[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
    
    cv2.imshow("Filtered Squares + AprilTags", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
