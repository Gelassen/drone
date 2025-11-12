import cv2
import numpy as np
from collections import deque

class PoseFilter:
    """Фильтр для сглаживания rvec и tvec между кадрами"""
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.rvec_prev = None
        self.tvec_prev = None

    def smooth(self, rvec, tvec):
        if self.rvec_prev is None:
            self.rvec_prev = rvec
            self.tvec_prev = tvec
            return rvec, tvec
        rvec_s = self.alpha*rvec + (1-self.alpha)*self.rvec_prev
        tvec_s = self.alpha*tvec + (1-self.alpha)*self.tvec_prev
        self.rvec_prev = rvec_s
        self.tvec_prev = tvec_s
        return rvec_s, tvec_s


class MarkerVisualizer:
    """Отрисовка маркеров и осей"""
    @staticmethod
    def drawAxesSafe(frame, camera_matrix, dist_coeffs, rvec, tvec, length=0.05):
        try:
            if hasattr(cv2, "drawFrameAxes"):
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, length)
            elif hasattr(cv2.aruco, "drawAxis"):
                cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, length)
        except Exception:
            # ручная отрисовка
            R, _ = cv2.Rodrigues(rvec)
            axis_points = np.float32([[0,0,0],[length,0,0],[0,length,0],[0,0,length]])
            proj, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
            proj = proj.reshape(-1,2).astype(int)
            cv2.line(frame, tuple(proj[0]), tuple(proj[1]), (0,0,255),2)
            cv2.line(frame, tuple(proj[0]), tuple(proj[2]), (0,255,0),2)
            cv2.line(frame, tuple(proj[0]), tuple(proj[3]), (255,0,0),2)

    @staticmethod
    def drawMarkers(frame, corners, ids, rejected=None):
        if ids is not None and len(ids)>0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        if rejected is not None and len(rejected)>0:
            cv2.aruco.drawDetectedMarkers(frame, rejected, borderColor=(128,0,128))


class ArUcoDetector:
    """Класс для детекции и оценки позы ArUco маркеров"""
    def __init__(self, camera_matrix, dist_coeffs, marker_length=0.1,
                 dictionary=cv2.aruco.DICT_5X5_50, alpha=0.5):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.marker_length = marker_length
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
        self.parameters = self.__create_detector_parameters() 
        self.parameters.adaptiveThreshWinSizeMin = 3
        self.parameters.adaptiveThreshWinSizeMax = 23
        self.parameters.adaptiveThreshWinSizeStep = 2
        self.parameters.adaptiveThreshConstant = 7
        self.parameters.minMarkerPerimeterRate = 0.01
        self.parameters.maxMarkerPerimeterRate = 4.0
        self.parameters.polygonalApproxAccuracyRate = 0.03
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.parameters.cornerRefinementWinSize = 5
        self.parameters.cornerRefinementMaxIterations = 30
        self.parameters.cornerRefinementMinAccuracy = 0.01
        self.pose_filter = PoseFilter(alpha=alpha)

    def __create_detector_parameters(self):
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
        gray = self.preProcess(frame)
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.parameters
        )
        return corners, ids, rejected

    def estimatePose(self, corners, ids):
        poses = []
        if ids is not None and len(ids)>0:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_length, self.camera_matrix, self.dist_coeffs
            )
            for rvec, tvec in zip(rvecs, tvecs):
                rvec_s, tvec_s = self.pose_filter.smooth(rvec, tvec)
                poses.append((rvec_s, tvec_s))
        return poses

    def postProcess(self, frame, corners, ids, rejected=None):
        """Отрисовка маркеров и осей"""
        MarkerVisualizer.drawMarkers(frame, corners, ids, rejected)
        poses = self.estimatePose(corners, ids)
        for rvec, tvec in poses:
            MarkerVisualizer.drawAxesSafe(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec)
        return frame, poses
