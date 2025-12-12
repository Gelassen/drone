#!/usr/bin/env python

import time
import math
import datetime
from pymavlink import mavutil
import cv2 as cv
import numpy as np

class OpticalFlowNode:
    def __init__(self):
        #self.vehicle = dronekit.connect(self.vehicle_connection_string, wait_ready=True, baud=self.vehicle_baudrate)
        self.mavlink_connection = mavutil.mavlink_connection('0.0.0.0:14445', autoreconnect=True, baud=57600)
        #self.mavlink_connection.wait_heartbeat()

        self.sensor_id = 64
        self.prev_img = None
        self.prev_features = None
        self.img = None
        self.features = None

        # Parameters for Shi-Tomasi corner detection
        self.feature_params = dict(maxCorners=100, qualityLevel=0.2, minDistance=2, blockSize=7)
        # Parameters for Lucas-Kanade optical flow
        self.lk_params = dict(winSize=(15,15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS|cv.TERM_CRITERIA_COUNT, 10, 0.03))

        # Hanning window for Phase correlation
        self.array = cv.createHanningWindow((480, 320), cv.CV_32F)
        #self.array = cv.createHanningWindow((1296, 1024), cv.CV_32F)


    def image(self, img):
        if self.prev_img is None:
            self.prev_img = img.copy()
            self.prev_features = cv.goodFeaturesToTrack(self.prev_img, mask=None, **self.feature_params)
        else:
            self.img = img.copy()
            #pix_x, pix_y, confidence = self.calculate_pix()
            pix_x, pix_y, confidence = self.calculate_phase_corr()
            self.send_mavlink_msg(pix_x, pix_y, confidence)
            self.prev_img = img.copy()
    

    def calculate_phase_corr(self):
        """
            Optical flow using phase correlation analogous to OpenMV sensor
        """
        shift, response = cv.phaseCorrelate(self.prev_img.astype('float32'), self.img.astype('float32'), window=self.array)
        #shift, response = cv.phaseCorrelate(self.prev_img.astype('float32'), self.img.astype('float32'))
        return shift[0], shift[1], response


    def calculate_pix(self):
        """
            Compute average optical flow in x and y in pixels
            Confidence is the number of total tracked features with good status==1
        """
        self.features, self.status, self.error = cv.calcOpticalFlowPyrLK(
            self.prev_img, self.img, self.prev_features, None, **self.lk_params)
        if self.status.any():
            # TODO: compare mean and mode for stability
            pix_x = int((self.features[:,:,0][self.status==1]-self.prev_features[:,:,0][self.status==1]).mean())
            pix_y = int((self.features[:,:,1][self.status==1]-self.prev_features[:,:,1][self.status==1]).mean())
            confidence = int(self.status.sum() / len(self.status) * 255.)
        else:
            pix_x, pix_y, confidence = 0, 0, 0
        # re-track new features to avoid cheking how many old features are still valid
        corners = cv.goodFeaturesToTrack(self.img, mask=None, **self.feature_params)
        #print(corners)
        if corners is not None:
            self.prev_features = corners
        print(.5, 'OF: {} {} {}'.format(pix_x, pix_y, confidence))
        return pix_x, pix_y, confidence

    def send_mavlink_msg(self, pix_x, pix_y, confidence):
        #msg = self.vehicle.message_factory.optical_flow_encode(
        #    self.img_usec,
        #    self.sensor_id,
        #    pix_x, pix_y,
        #    0., 0.,
        #    confidence,
        #    -1.)
        print(int(time.time()*1000), self.sensor_id, pix_x, pix_y, 0, 0, confidence, 0)
        #self.mavlink_connection.mav.optical_flow_send(1, 100, 1, 1, 0, 0, 0, 1)
        #self.mavlink_connection.mav.optical_flow_send(int(time.time()*1000), self.sensor_id, int(pix_x*100), int(pix_y*100), 0, 0, int(confidence), 0)
        self.mavlink_connection.mav.optical_flow_send(int(time.time()*1000), self.sensor_id, int(pix_x), int(pix_y), 0, 0, int(confidence), 0)

if __name__ == '__main__':
    of = OpticalFlowNode()

    cap = cv.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        #cv.imshow("", frame)
        
        #of.image(frame)
        of.image(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))
        #cv.waitKey(1)

