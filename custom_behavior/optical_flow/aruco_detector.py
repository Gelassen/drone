#!/usr/bin/env python3
import cv2
import numpy as np
import signal
import sys
from collections import deque

# ------------------------------
# PoseFilter: сглаживание только tvec
# ------------------------------
class PoseFilter:
    """Фильтр для сглаживания tvec между кадрами (rvec не усредняем, т.к. Rodrigues вектор нельзя линейно усреднять)"""
    def __init__(self, alpha=0.5):
        self.alpha = float(alpha)
        self.tvec_prev = None

    def smooth(self, rvec, tvec):
        # rvec возвращаем неизменным; tvec сглаживаем скользящим средним
        if self.tvec_prev is None:
            self.tvec_prev = tvec.copy()
            return rvec, tvec
        tvec_s = self.alpha * tvec + (1.0 - self.alpha) * self.tvec_prev
        self.tvec_prev = tvec_s
        return rvec, tvec_s

# ------------------------------
# MarkerVisualizer: отрисовки
# ------------------------------
class MarkerVisualizer:
    """Отрисовка маркеров, rejected, осей и подсветки"""
    @staticmethod
    def drawAxesSafe(frame, camera_matrix, dist_coeffs, rvec, tvec, length=0.05):
        try:
            # OpenCV >= 4.7
            if hasattr(cv2, "drawFrameAxes"):
                # signature: cv2.drawFrameAxes(image, cameraMatrix, distCoeffs, rvec, tvec, length)
                # cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, length)
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec[0], tvec[0], 0.05)
                return
            # older contrib
            if hasattr(cv2.aruco, "drawAxis"):
                cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, length)
                return
        except Exception:
            pass

        # fallback: ручная проекция трёх осей
        try:
            axis_points = np.float32([[0,0,0],[length,0,0],[0,length,0],[0,0,length]])
            proj, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
            proj = proj.reshape(-1,2).astype(int)
            cv2.line(frame, tuple(proj[0]), tuple(proj[1]), (0,0,255), 2)   # X - красный
            cv2.line(frame, tuple(proj[0]), tuple(proj[2]), (0,255,0), 2)   # Y - зелёный
            cv2.line(frame, tuple(proj[0]), tuple(proj[3]), (255,0,0), 2)   # Z - синий
        except Exception:
            # ничего не делаем, если проекция упала
            pass

    @staticmethod
    def drawMarkers(frame, corners, ids, rejected=None):
        if ids is not None and len(ids) > 0:
            try:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            except Exception:
                # fallback: рисуем полигоны вручную
                for c in corners:
                    poly = c.reshape(-1,2).astype(int)
                    cv2.polylines(frame, [poly], True, (0,255,0), 2)
        if rejected is not None and len(rejected) > 0:
            try:
                cv2.aruco.drawDetectedMarkers(frame, rejected, borderColor=(128,0,128))
            except Exception:
                for c in rejected:
                    poly = c.reshape(-1,2).astype(int)
                    cv2.polylines(frame, [poly], True, (128,0,128), 1)

# ------------------------------
# ArUcoDetector: главный класс
# ------------------------------
class ArUcoDetector:
    """Класс для детекции и оценки позы ArUco маркеров"""
    def __init__(self, camera_matrix, dist_coeffs,
                 marker_length=0.10,
                 dictionary=cv2.aruco.DICT_5X5_50,
                 alpha=0.5):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.marker_length = float(marker_length)
        # словарь маркеров (по умолчанию один, можно расширить)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)

        # параметры детектора
        self.parameters = self.__create_detector_parameters()
        # tuned params (рекомендации)
        self.parameters.adaptiveThreshWinSizeMin = 3
        self.parameters.adaptiveThreshWinSizeMax = 23
        self.parameters.adaptiveThreshWinSizeStep = 2
        self.parameters.adaptiveThreshConstant = 3      # уменьшено (было 7)
        self.parameters.minMarkerPerimeterRate = 0.02   # чуть выше, чтобы исключить мегашум
        self.parameters.maxMarkerPerimeterRate = 1.0
        self.parameters.polygonalApproxAccuracyRate = 0.05
        # corner refinement для точности
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.parameters.cornerRefinementWinSize = 5
        self.parameters.cornerRefinementMaxIterations = 30
        self.parameters.cornerRefinementMinAccuracy = 0.01

        # фильтр для tvec
        self.pose_filter = PoseFilter(alpha=alpha)

    def __create_detector_parameters(self):
        # поддержка разных API
        if hasattr(cv2.aruco, "DetectorParameters"):
            return cv2.aruco.DetectorParameters()
        else:
            return cv2.aruco.DetectorParameters_create()

    def preProcess(self, frame):
        """CLAHE + GaussianBlur для устойчивости к бликам и шуму"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        gray = cv2.GaussianBlur(gray, (3,3), 0)
        return gray

    def detect(self, frame):
        """Детекция: возвращаем corners, ids, rejected — и используем processed gray"""
        gray = self.preProcess(frame)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
        return corners, ids, rejected

    def estimatePose(self, corners, ids):
        """Оценка позы для всех найденных маркеров. Возвращаем список (id, rvec, tvec)"""
        poses = []
        if ids is None or len(ids) == 0:
            return poses

        # rvecs, tvecs shapes: (N,1,3)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_length,
                                                              self.camera_matrix, self.dist_coeffs)
        for idx, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            # rvec, tvec имеют формы (1,3) или (3,) в зависимости от версии — унифицируем
            r = rvec.reshape(3,1) if rvec.shape != (3,1) else rvec
            t = tvec.reshape(3,1) if tvec.shape != (3,1) else tvec
            # smooth только tvec
            r_s, t_s = self.pose_filter.smooth(r, t)
            poses.append((int(ids[idx][0]), r_s, t_s))
        return poses

    def postProcess(self, frame, corners, ids, rejected=None, debug=False):
        """Отрисовка маркеров, rejected, осей; возвращаем frame и poses"""
        # draw all markers (detected & rejected)
        MarkerVisualizer.drawMarkers(frame, corners if corners is not None else [], ids, rejected)

        poses = self.estimatePose(corners, ids)
        pose_ids = {p[0] for p in poses}

        # draw axes for those poses; highlight markers without pose
        if ids is not None:
            for i, c in enumerate(corners):
                mid_poly = c.reshape(-1,2).astype(int)
                marker_id = int(ids[i][0])
                if marker_id in pose_ids:
                    # find pose entry
                    entry = next(filter(lambda x: x[0] == marker_id, poses))
                    _, r_s, t_s = entry
                    MarkerVisualizer.drawAxesSafe(frame, self.camera_matrix, self.dist_coeffs, r_s, t_s, length=self.marker_length*0.5)
                else:
                    # подсветка маркера, для которого pose не рассчитана
                    cv2.polylines(frame, [mid_poly], True, (0,0,255), 2)

        # debug overlay
        if debug:
            det_count = 0 if ids is None else len(ids)
            rej_count = 0 if rejected is None else len(rejected)
            cv2.putText(frame, f"Detected:{det_count} Rejected:{rej_count}", (10,20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        return frame, poses