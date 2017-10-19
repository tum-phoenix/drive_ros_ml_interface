#!/usr/bin/env python
import sys
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os
import numpy as np
import pickle
from keras.models import load_model

class TrafficSignPreProcssing:
    def __init__(self):
        self.opencv_debug = rospy.get_param('~opencv_debug', 10.0)

        # debug parameters
        self.sigmacolor = 175
        self.sigmaspace = 175
        self.bilateral_diameter = 5
        self.lower_canny_thresh = 95
        self.upper_canny_thresh = 430
        self.dp = 1
        self.minDist = 100
        self.param1 = 50
        self.param2 = 30
        self.minRadius = 25
        self.maxRadius = 50

        if self.opencv_debug:
            cv2.namedWindow('setBilateral')
            cv2.namedWindow('setCanny')
            cv2.namedWindow('houghCircles')
            # canny trackbars
            cv2.createTrackbar('lower','setCanny',self.lower_canny_thresh,1000,self.lower_callback)
            cv2.createTrackbar('upper','setCanny',self.upper_canny_thresh,1000,self.upper_callback)
            # bilateral filter trackbars
            cv2.createTrackbar('diameter','setBilateral',self.bilateral_diameter,20,self.bilateral_diameter_callback)
            cv2.createTrackbar('sigmacolor','setBilateral',self.sigmacolor,1000,self.sigmacolor_callback)
            cv2.createTrackbar('sigmaspace','setBilateral',self.sigmaspace,1000,self.sigmaspace_callback)
            # hough circle trackbars
            cv2.createTrackbar('dp','houghCircles',self.dp,20,self.dp_callback)
            cv2.createTrackbar('minDist','houghCircles',self.minDist,1000,self.minDist_callback)
            cv2.createTrackbar('param1','houghCircles',self.param1,1000,self.param1_callback)
            cv2.createTrackbar('param2','houghCircles',self.param2,1000,self.param2_callback)
            cv2.createTrackbar('minRadius','houghCircles',self.minRadius,1000,self.minRadius_callback)
            cv2.createTrackbar('maxRadius','houghCircles',self.maxRadius,1000,self.maxRadius_callback)

        self.model = load_model(rospy.get_param('~net_model'))
        #self.names_dict = pickle.Unpickler(open(rospy.get_param('~signs_dict'))).load()
        # the notebook uses python3, python2 pickle cannot read it so we will simply generate this
        self.names_dict = {'0': 'Geschwindigkeitsbegrenzung 20', '1': 'Geschwindigkeitsbegrenzung 30', '2': 'Geschwindigkeitsbegrenzung 50', '3': 'Geschwindigkeitsbegrenzung 60', '4': 'Geschwindigkeitsbegrenzung 70', '5': 'Geschwindigkeitsbegrenzung 80', '6': 'Geschwindigkeitsbegrenzung Ende 80', '7': 'Geschwindigkeitsbegrenzung 100', '8': 'Geschwindigkeitsbegrenzung 120', '9': 'Ueberholverbot PKW', '10': 'Ueberholverbot LKW', '11': 'Naechste Kreuzung Vorfahrt', '12': 'Vorfahtsstrasse', '13': 'Vorfahrt gewaehren', '14': 'STOP', '15': 'Durchfahrt verboten', '16': 'Durchfahrt verboten LKWs', '17': 'Einbahnstrasse falsche Seite', '18': '!-Zeichen', '19': 'Scharfe Kurve links', '20': 'Scharfe Kurve rechts', '21': 'Kurvige Strasse', '22': 'Bodenwellen', '23': 'Rutschgefahr', '24': 'Strasse wird enger links', '25': 'Vorsicht Bauarbeiten', '26': 'Vorsicht Ampel', '27': 'Vorsicht Fussgaenger', '28': 'Vorsicht Kinder', '29': 'Vorsicht Fahrrad', '30': 'Vorsicht Schnee', '31': 'Vorsicht Wild', '32': 'Alle Regeln frei', '33': 'Rechts Abbiegen', '34': 'Links Abbiegen', '35': 'Geradeaus', '36': 'Geradeaus oder rechts abbiegen', '37': 'Geradeaus oder links abbiegen', '38': 'Blauer Pfeil untenrechts', '39': 'Blauer Pfeil untenlinks', '40': 'Kreisverkehr', '41': 'Ueberholverbot aufgehoben', '42': 'LKW-Ueberholverbot aufgehoben', '43': 'Nothing'} 
        self.bridge = CvBridge()
        # threshold for contours below which they will be discarded
        self.contour_area_threshold = rospy.get_param('~contour_threshold', 10.0)
        # inflate a bit to simplify processing
        self.contour_inflation = rospy.get_param('~contour_inflation',4)
        if (rospy.get_param("~/use_topic", False) == False):
            rospy.loginfo("Traffic sign preprocessing node initialized - reading from video")
            self.cap = cv2.VideoCapture(rospy.get_param('~video_filepath'))
            self.read_video()
        else:
            rospy.loginfo("Traffic sign preprocessing node initialized - listening for subscription")
            self.image_sub = rospy.Subscriber('img_in', Image, self.imageCallback)

    def read_video(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES,1500)
        while (self.cap.isOpened()):
            ret, frame = self.cap.read()
            if frame is not None:
                self.processImage(frame)

    # debug opencv slider callbacks
    def lower_callback(self, lower):
        self.lower_canny_thresh = lower

    def upper_callback(self, upper):
        self.upper_canny_thresh = upper

    def sigmacolor_callback(self, sigmacolor):
        self.sigmacolor = sigmacolor

    def sigmaspace_callback(self, sigmaspace):
        self.sigmaspace = sigmaspace
    
    def bilateral_diameter_callback(self, bilateral_diameter):
        self.bilateral_diameter = bilateral_diameter

    def dp_callback(self, dp):
        self.dp = dp

    def minDist_callback(self, minDist):
        self.minDist = minDist

    def param1_callback(self, param1):
        self.param1 = param1

    def param2_callback(self, param2):
        self.param2 = param2

    def minRadius_callback(self, minRadius):
        self.minRadius = minRadius

    def maxRadius_callback(self, maxRadius):
        self.maxRadius = maxRadius
        
    def imageCallback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data,"bgr8")
        except CvBridgeError as e:
            print(e)

        rospy.loginfo("Image callback received")
        cv2.imshow("Received image", cv_image)
        cv2.waitKey(1)       
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        self.processImage(gray)

    # debug openCV windows to adjust parameters
    def processImage(self, image):

        image = image[200:400, 400:640]
        cv2.imshow('Input image', image)

        if self.opencv_debug:
            draw_image = image.copy()

        bilateral_filtered_image = cv2.bilateralFilter(image, self.bilateral_diameter, self.sigmacolor, self.sigmaspace)
        
        if self.opencv_debug:
            cv2.imshow('setBilateral', bilateral_filtered_image)
        edge_detected_image = cv2.Canny(bilateral_filtered_image, self.lower_canny_thresh, self.upper_canny_thresh)
        
        # if self.opencv_debug:
        cv2.imshow('setCanny',edge_detected_image)

        circles = cv2.HoughCircles(edge_detected_image,cv2.HOUGH_GRADIENT,self.dp,self.minDist,
                            param1=self.param1,param2=self.param2,minRadius=self.minRadius,maxRadius=self.maxRadius)

        interest_regions = []
        if circles is not None:
            image_draw_circles = image.copy()
            circles = np.uint16(np.around(circles))
            for i in circles[0,:]:
                x_lower = int(i[0])-int(i[2])
                if x_lower < 0:
                    x_lower = 0
                y_lower = int(i[1])-int(i[2])
                if y_lower < 0:
                    y_lower = 0
                interest_regions.append(cv2.resize(image_draw_circles.copy()[x_lower:i[0]+i[2],y_lower:i[1]+i[2]],dsize=(64,64)))
                # draw the outer circle
                cv2.circle(image_draw_circles,(i[0],i[1]),i[2],(0,255,0),2)
                # draw the center of the circle
                cv2.circle(image_draw_circles,(i[0],i[1]),2,(0,0,255),3)
            cv2.imshow('houghCircles',image_draw_circles)

        image_draw_cnts = image.copy()
        # cnt_img, cnts, hierarchy = cv2.findContours(edge_detected_image,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        #cnt_img, cnts, hierarchy = cv2.findContours(edge_detected_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cnt_img, cnts, hierarchy = cv2.findContours(edge_detected_image,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

        # prev next something something       
        approx_cnts = []
        approx_cnts_idx = {}
        # print('Hierarchy: ',hierarchy)
        # print('at 121',hierarchy[0][1][0])

        for idx, cnt in enumerate(cnts):
            # todo: check if hierarchy usage is valid
            if len(cnt) == 2 or approx_cnts_idx.has_key(hierarchy[0][idx][3]):
                approx_cnts_idx[idx] = True
                continue
            epsilon = 0.1*cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,epsilon,True)
            if not cv2.isContourConvex(approx):
                continue
            area = cv2.contourArea(cnt)
            if len(approx) < 6 and area > 200:
                # print('Inserting:', idx)
                # print('Hierarchy of it: ', hierarchy[0][idx])
                approx_cnts.append(approx)
                approx_cnts_idx[idx] = True
                # print('Excluded: ', approx_cnts_idx)
                x,y,w,h = cv2.boundingRect(approx)
                w_old = w
                h_old = h
                w = int(w+w_old*self.contour_inflation)
                h = int(h+h_old*self.contour_inflation)
                x = int(x-0.5*w_old*self.contour_inflation)
                if x < 0:
                    x = 0
                y = int(y-0.5*h_old*self.contour_inflation)
                if y < 0:
                    y = 0
                interest_regions.append(cv2.resize(image_draw_cnts.copy()[y:y+h,x:x+h],dsize=(64,64)))

        cv2.drawContours(image_draw_cnts, approx_cnts, -1, (0, 255, 0), 3)
        cv2.imshow('Detected contours', image_draw_cnts)

        # for contour in interest_regions:
        #     if contour is not None and contour.size:
        #         cv2.imshow("Potential sign",contour)
        
        for region in interest_regions:
            predictions = self.model.predict(np.array([region]), verbose=1)
            for idx,prediction in enumerate(predictions):
                print("Prediction:",np.argmax(predictions[idx]),"meaning sign:",self.names_dict[str(np.argmax(predictions[idx]))])
                cv2.imshow("Image",interest_regions[idx])
                cv2.waitKey(0)

def main(args):
    rospy.init_node('traffic_sign_preprocessing', anonymous=True)
    traffic_sign_preprocessing = TrafficSignPreProcssing()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    cv2.destroyAllWindows()
    traffic_sign_preprocessing.cap.release()

if __name__ == '__main__':
    main(sys.argv)