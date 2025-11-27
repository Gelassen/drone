import cv2
import numpy as np
from pupil_apriltags import Detector


# ----------------------------
#  1) Инициализация AprilTag
# ----------------------------
at_detector = Detector(
    families="tag36h11",
    nthreads=4,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=True,
    decode_sharpening=0.25,
    debug=False
)


# ----------------------------
#  2) Поиск квадратов на кадре
# ----------------------------
def find_square_candidates(frame, min_area=1500):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # поиск контуров
    edges = cv2.Canny(blurred, 40, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    squares = []

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if area < min_area:
                continue

            # Проверка "квадратности": отношение сторон 0.7–1.3
            x, y, w, h = cv2.boundingRect(approx)
            ratio = w / float(h)
            if 0.7 < ratio < 1.3:
                squares.append((x, y, w, h))

    return squares


# ----------------------------------------
#  3) Попытка декодировать AprilTag внутри
# ----------------------------------------
def detect_tag_in_square(frame, box):
    x, y, w, h = box
    roi = frame[y:y+h, x:x+w]

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    tags = at_detector.detect(gray_roi)

    results = []
    for tag in tags:
        # переведём координаты обратно в систему полного кадра
        tag_center = (int(tag.center[0] + x), int(tag.center[1] + y))
        results.append({
            "id": tag.tag_id,
            "center": tag_center,
            "corners": [(int(c[0] + x), int(c[1] + y)) for c in tag.corners]
        })
    return results


# ----------------------------
#  4) Основной цикл
# ----------------------------
def main():
    video_path = "../../assets/ar_test_video.MOV"
    cap = cv2.VideoCapture(video_path)  # поменяй на свой источник

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        squares = find_square_candidates(frame)

        for box in squares:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

            tags = detect_tag_in_square(frame, box)
            for t in tags:
                cv2.circle(frame, t["center"], 4, (0, 0, 255), -1)
                cv2.putText(frame, f"ID {t['id']}", t["center"],
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                for cx, cy in t["corners"]:
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

        cv2.imshow("Squares + AprilTag", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break



if __name__ == "__main__":
    main()
