#!/usr/bin/env python
# Note that this needs:
# sudo pip install websocket-client
# not the library called 'websocket'

'''
MIT License

Copyright (c) 2019 Stephen Vu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

from lane_detection import laneDetection
from numpy import linalg as LA
import numpy as np
from utils import *
import math
import cv2

class steering_angle:

    def __init__(self,frame,lines,transform_matrix):
        
        self.frame = frame
        self.lines = lines
        self.transform_matrix = transform_matrix
        self.lane_width = 90
    def cal_slope(self,x1, y1, x2, y2):
        self.slope = float(0)
        self.b = float((x2 - x1))
        self.d = float((y2 - y1))
        if self.d != 0:
            self.slope = (self.b)/(self.d) 

        return self.slope

    def average_slop_intercept(self):
        """
        This function combines line segments into one or two lane lines
        If all line slopes are < 0: then we only have detected left lane
        If all line slopes are > 0: then we only have detected right lane
        """

        self.lane_lines = []

        if self.lines is None:
            #rospy.logwarn('No line_segment segments detected')
            return self.lane_lines 

        self.max_norm_right=0
        self.max_norm_left=0
        self.coordinates_right_max=[]
        self.coordinates_left_max=[]
        #print("Line segments: ",len(line_segments))
        self.line_segment = None
        try:
            for self.line_segment in self.lines:
                #print(line_segment)
                for x1, y1, x2, y2 in self.line_segment:

                    self.slope= self.cal_slope(x1,y1,x2,y2)
                    angle= float(math.atan(self.slope))
                    self.angle_degree= float(angle * 180.0 / math.pi)

                    if self.slope < 0:
                        if y1==y2:
                            continue
                        """
                        if x1 < left_region_boundary and x2 < left_region_boundary:
                            left_fit.append((slope, intercept))
                        """
                        #print('left_slope', slope)
                        coordinates_right = np.array([[x1,y1,x2,y2]])
                        norm_right = LA.norm(coordinates_right)
                        if norm_right > self.max_norm_right and self.angle_degree >= -70:# and self.slope >= -1.7:
                            #print (slope)
                            self.max_norm_right=norm_right
                            self.coordinates_right_max=coordinates_right 
                    else:
                        if y1==y2:
                            continue                        
                        """
                        if x1 > right_region_boundary and x2 > right_region_boundary:
                            right_fit.append((slope, intercept))
                        """
                        #print('right_slope',slope)
                        coordinates_left = np.array([[x1,y1,x2,y2]])

                        norm_left = LA.norm(coordinates_left)
                        if norm_left > self.max_norm_left and self.angle_degree <= 70:# and self.slope <=1.7:
                            self.max_norm_left=norm_left
                            self.coordinates_left_max=coordinates_left
        except:
            pass
            #print("No line")
        self.lane_lines.append(self.coordinates_left_max)
        self.lane_lines.append(self.coordinates_right_max)
        return self.lane_lines 
    # ################################### TEST FOR WHITE NOISE #############################        
    # def average_slop_intercept_noise(self,last_check_point):
    #     """
    #     This function combines line segments into one or two lane lines
    #     If all line slopes are < 0: then we only have detected left lane
    #     If all line slopes are > 0: then we only have detected right lane
    #     """

    #     self.lane_lines = []

    #     if self.lines is None:
    #         print('No line_segment segments detected')
    #         return self.lane_lines 

    #     self.max_norm_right=0
    #     self.max_norm_left=0
    #     self.coordinates_right_max=[]
    #     self.coordinates_left_max=[]
    #     #print("Line segments: ",len(line_segments))
    #     self.line_segment = None
    #     try:
    #         for self.line_segment in self.lines:
    #             #print(line_segment)
    #             for x1, y1, x2, y2 in self.line_segment:

    #                 self.slope= self.cal_slope(x1,y1,x2,y2) # Neu vector nhan voi goc cua o thoi diem cuoi truoc khi vao nhieu > 0 suy ra lay vector
    #                 angle= float(math.atan(self.slope))
    #                 self.angle_degree= float(angle * 180.0 / math.pi)
    #                 if y1==y2:
    #                     continue
    #                 if - self.angle_degree * last_check_point > 0 and 0<abs(self.angle_degree)<abs(last_check_point): # Goc cua ma nhan voi nhau > 0 suy ra cung huong voi nhau
    #                     print('Noise ', - self.angle_degree)
    #                     coordinates_right = np.array([[x1,y1,x2,y2]])
    #                     norm_right = LA.norm(coordinates_right) 
    #                     if norm_right > self.max_norm_right:
    #                         #print (slope)
    #                         self.max_norm_right=norm_right
    #                         self.coordinates_right_max=coordinates_right 
    #     except:
    #         print("No line")
    #     self.lane_lines.append(self.coordinates_right_max)
    #     return self.lane_lines 

    def display(self):

        self.lane_lines_image = display_lines(self.frame,self.lane_lines) # Form utils with love
        return self.lane_lines_image

    def compute_angle_and_speed(self):
        
        #print(lane_lines)
        #print(len(lane_lines[0]))
        #print(len(lane_lines[1]))
        # boundary = 1/3
        # left_region_boundary = 210  # left lane line segment should be on left 2/3 of the screen
        # right_region_boundary = 110 #320 * boundary # right lane line segment should be on left 2/3 of the screen
        # check_lane = 0 # 0 is middle, -1 is left lane, 1 is right lane
        if len(self.lane_lines[0]) == 1 and len(self.lane_lines[1]) == 1:

            '''
            Can phai tinh gia tri cua lanewidth 
            vi moi duong co gia tri do dai khac nhau
            norm (v1,v2) --> Thuc nghiem de cho ra 3 loai duong khac nhau ( duong nho, duong trung binh, va duong to)
            duong nho thuong kho di (vi goc lai lon)
            Thuc nghiem cho thay gia tri do dai lanewidth doi voi duong to nam trong khoang (300,360)
            '''

            #print("Middle lane")

            #print(lane_lines[0][0])
            left_x1, left_y1, left_x2, left_y2 = self.lane_lines[0][0]
            #print(left_x1, left_y1, left_x2, left_y2)
            right_x1, right_y1, right_x2, right_y2 = self.lane_lines[1][0]
            #print(right_x1, right_y1, right_x2, right_y2)

            #################### TEST BIRD VIEW FOR 4 POINTS ################
            #points_to_be_transformed = np.array([[[left_x1, left_y1],[[left_x2, left_y2]],[right_x1, right_y1],[right_x2, right_y2]]], dtype=np.float32)#diem chuyen sang birdview
            points_to_be_transformed = np.array([[[left_x1, left_y1],[left_x2, left_y2],[right_x1, right_y1],[right_x2, right_y2]]], dtype=np.float32)#diem chuyen sang birdview
            bird_view = cv2.perspectiveTransform(points_to_be_transformed, self.transform_matrix['M'])


            #print("ABC",abc[0][0])  
            #       
            left_x1_bird_view = bird_view[0][0][0]
            left_y1_bird_view = bird_view[0][0][1]
            left_x2_bird_view = bird_view[0][1][0]
            left_y2_bird_view = bird_view[0][1][1]
            right_x1_bird_view = bird_view[0][2][0]
            right_y1_bird_view = bird_view[0][2][1]
            right_x2_bird_view = bird_view[0][3][0]
            right_y2_bird_view = bird_view[0][3][1]

            mid_x_left = (left_x1_bird_view + left_x2_bird_view) / float(2)
            mid_y_left = (left_y1_bird_view + left_y2_bird_view) / float(2)
            mid_x_right = (right_x1_bird_view + right_x2_bird_view) / float(2)
            mid_y_right = (right_y1_bird_view + right_y2_bird_view) / float(2)

            self.lane_width = math.sqrt((mid_x_left - mid_x_right)**2 + (mid_y_left - mid_y_right)**2)
          
            mid_x = (mid_x_left + mid_x_right) / float(2)
            mid_y = (mid_y_left + mid_y_right) / float(2)
            
            slope_mid = (mid_x-160) / float(mid_y -240)
            angle_radian_mid= float(math.atan(slope_mid))
            self.angle_degree_mid= float(angle_radian_mid * 180.0 / math.pi)

            #print("Gia tri bird view: ", -self.angle_degree_mid)
            # Tinh norm khoang cach duong 

            self.check_lane = 0
            return self.angle_degree_mid,self.check_lane,self.lane_width #Old = angle_degree

        if len(self.lane_lines[0]) == 1 and len(self.lane_lines[1]) == 0:

            # Detect right line
            
            #print("Right lane")
            self.check_lane = 1
            #print (lane_lines)
            x1,y1,x2,y2= self.lane_lines[0][0]
            #print(x1,y1,x2,y2)
            #slope = float((x1-x2)/(y1-y2))
            slope = (x1-x2) / float(y1 -y2)
            
            ################# TEST BIRD VIEW FOR ONE LANE #######################
            points_to_be_transformed = np.array([[[x1, y1],[x2, y2]]], dtype=np.float32)#diem chuyen sang birdview
            bird_view = cv2.perspectiveTransform(points_to_be_transformed, self.transform_matrix['M'])
            
            right_x1_bird_view = bird_view[0][0][0]
            right_y1_bird_view = bird_view[0][0][1]
            right_x2_bird_view = bird_view[0][1][0]
            right_y2_bird_view = bird_view[0][1][1]

            # Ve canh ao bang viec dich chuyen toa do voi do dai lane_width
            
            #left_x_mid = (left_x1_bird_view + left_x2_bird_view ) / float(2)
            #left_y_mid = (left_y1_bird_view + left_y2_bird_view ) / float(2)

            slope_bird_view = (right_x1_bird_view-right_x2_bird_view) / float(right_y1_bird_view -right_y2_bird_view)
            angle_radian_predict= float(math.atan(slope_bird_view))
            self.angle_degree_predict= float(angle_radian_predict * 180.0 / math.pi)
            self.lane_width = 90
            #print("goc du doan doi voi right lane: ",-self.angle_degree_predict)
            return self.angle_degree_predict,self.check_lane,self.lane_width #angle_degree
            
        if len(self.lane_lines[0]) == 0 and len(self.lane_lines[1]) == 1:
            # Do something
            # Lane trai 
            #print('Left lane')
            self.check_lane = -1
            #print(lane_lines)
            
            #print("1")
            #print (lane_lines)
            x1,y1,x2,y2= self.lane_lines[1][0]

            ################# TEST BIRD VIEW FOR ONE LANE #######################
            points_to_be_transformed = np.array([[[x1, y1],[x2, y2]]], dtype=np.float32)#diem chuyen sang birdview
            bird_view = cv2.perspectiveTransform(points_to_be_transformed, self.transform_matrix['M'])
            
            left_x1_bird_view = bird_view[0][0][0]
            left_y1_bird_view = bird_view[0][0][1]
            left_x2_bird_view = bird_view[0][1][0]
            left_y2_bird_view = bird_view[0][1][1]

            # Ve canh ao bang viec dich chuyen toa do voi do dai lane_width
            
            #left_x_mid = (left_x1_bird_view + left_x2_bird_view ) / float(2)
            #left_y_mid = (left_y1_bird_view + left_y2_bird_view ) / float(2)

            slope_bird_view = (left_x1_bird_view-left_x2_bird_view) / float(left_y1_bird_view -left_y2_bird_view)

            angle_radian_predict= float(math.atan(slope_bird_view))

            self.angle_degree_predict= float(angle_radian_predict * 180.0 / math.pi)

            #print("Goc du doan doi voi left lane:",-self.angle_degree_predict)
            self.lane_width = 90
            return self.angle_degree_predict,self.check_lane,self.lane_width #old: angree_degree
                    
