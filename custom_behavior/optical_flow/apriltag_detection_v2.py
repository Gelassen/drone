import cv2
import numpy as np
import apriltag

# --- AprilTag детектор ---
apriltag_detector = apriltag.Detector(apriltag.DetectorOptions(families="tag16h5"))

# --- Камера / видео ---
video_path = "../../assets/ar_test_video.MOV"
cap = cv2.VideoCapture(video_path)

# --- Параметры фильтрации квадратов ---
MIN_AREA = 2000
MAX_AREA = 15000
ASPECT_RATIO_TOL = 0.2

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

def draw_axes(frame, corners, scale=20):
    print("draw_axes is called")
    """Рисуем локальные 2D-оси маркера по его углам"""
    corners = corners.astype(int)
    center = corners.mean(axis=0).astype(int)
    
    # ось X — от центра к первому углу
    cv2.line(frame, tuple(center), tuple(corners[0]), (0,0,255), 2)
    # ось Y — от центра к следующему углу (по часовой стрелке)
    cv2.line(frame, tuple(center), tuple(corners[1]), (0,255,0), 2)
    # ось Z — для наглядности вертикальная линия через центр (синяя)
    cv2.line(frame, tuple(center), (center[0], center[1]-scale), (255,0,0), 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    squares = find_squares(frame)
    print("Inference is running")
    
    for (x, y, w, h) in squares:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,255), 2)
        roi = frame[y:y+h, x:x+w]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        detections = apriltag_detector.detect(gray_roi)
        
        print("detections length is ", len(detections))
        for det in detections:
            corners = det.corners + np.array([x, y])
            cv2.polylines(frame, [corners.astype(int)], isClosed=True, color=(0,255,0), thickness=2)
            cv2.putText(frame, f"ID: {det.tag_id}", tuple(corners[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
            
            draw_axes(frame, corners, scale=15)

    cv2.imshow("Filtered Squares + AprilTags + Axes", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
