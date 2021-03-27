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

import algorithms

from algorithms.utils import *
from algorithms.rosws import *
import math
from algorithms.lane_detection import laneDetection
from algorithms.steering_angle_processing import steering_angle
from algorithms.traffic_sign_detection import traffic_sign
from numpy import linalg as LA
from depth_processing import *
import pickle
import time


import message_filters

def callback_image(raw):
    np_arr = np.fromstring(raw.data, np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image_np

class traffic_task:
    
    def __init__(self):

        self.subscriber1 = rospy.Subscriber("g2_never_die/camera/rgb/compressed",
                                            CompressedImage,
                                            self.callback,
                                            queue_size=1)

        self.traffic_sign = rospy.Subscriber("g2_never_die/traffic_sign",
                                            Int8,
                                            self.callback_sign,
                                            queue_size=1)

        # self.balance = rospy.Subscriber("g2_never_die/balance",
        #                                    Int8,
        #                                    self.balance_lane,
        #                                   queue_size=1) 

        self.speed = rospy.Publisher("g2_never_die/set_speed",Float32,queue_size = 1)

        self.angle_car = rospy.Publisher("g2_never_die/set_angle",Float32,queue_size = 1)

        self.flag_sign = 0

        self.nga_tu = 0

        self.flag = 0 

        self.steering_bool = False

        self.count_for_traffic = 0

        self.balance_road = 0

    # def balance_lane(self,balance_data):

    #     rospy.logerr("THIS IS BALANCEEEEEEEEEEEEEE: " + str(balance_data))

    #     self.balance_road = balance_data.data
        
    def callback_sign(self,sign_data):
        
        rospy.logerr("THIS IS HEREEEEEEEEEEEEEEEEEEE: " + str(sign_data))
        self.flag_sign = sign_data.data

    def callback(self, ros_data):

        np_app = np.fromstring(ros_data.data,np.uint8)

        self.frame = cv2.imdecode(np_app,cv2.IMREAD_COLOR)

        cv2.imshow("Traffic",self.frame)
        
        self.traffic_detection_sign()

        cv2.waitKey(1)

    def traffic_detection_sign(self):

        lane_detection = laneDetection(self.frame)

        warp_img,gray_bird = lane_detection.processing_traffic_sign()         

        contours, _ = cv2.findContours(warp_img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]

        biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

        mask = np.zeros(self.frame.shape, np.uint8)

        cv2.drawContours(mask, [biggest_contour], -1, 255, -1)

        #cv2.imshow("Sign",mask)

        diem_re = 0

        try:
            #print("GLAFFFFFFFFFFFFFFF",flag)
            if self.flag_sign == -1:
                self.steering_bool = True
                for i in range(0,160):
                    if np.sum(mask[140][i]) == 255:
                        diem_re += 1
            elif self.flag_sign == 1:
                self.steering_bool = True
                for i in range(160,320):
                    if np.sum(mask[140][i]) == 255:
                        diem_re += 1 

            #print("Diem re: ", diem_re,self.flag_sign)

            if diem_re == 90:
                self.nga_tu = 1

            if self.nga_tu == 1:
                if diem_re < 90:
                    if self.flag_sign == -1:
                        rospy.loginfo("Left traffic sign detection!")
                        time_test = time.time()
                        #while steering_bool:

                        while time.time() - time_test <= 0.5: #time.time() - time_test <= 0.5steering_bool: #steering_bool: 

                            # lane =  connect.subscribe('g2_never_die/lane',Int8())

                            # print("LANEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE: ",lane.data)

                            # raw =  connect.subscribe('g2_never_die/camera/rgb/compressed', CompressedImage())

                            # frame = callback_image(raw)

                            # lane_choose = rospy.get_param('~lane_detection')

                            # fp_abc = open(lane_choose)

                            # shared_abc = pickle.load(fp_abc)

                            # lane = shared_abc["Lane"]

                            #print("ERROR: ",lane)

                            #flag = 1
                            #rospy.loginfo("This time to port to left!")
                            angle= Float32()
                            speed_car = Float32()
                            angle.data = -90
                            speed_car.data = 0
                            self.speed.publish(speed_car)
                            self.angle_car.publish(angle)
                            #connect.publish('g2_never_die/set_speed',speed)
                            #connect.publish('g2_never_die/set_angle',angle)   

                            # if self.balance_road != 0:
                            #     self.count_for_traffic += 1
                            #     if self.count_for_traffic == 30:
                            #         self.steering_bool = False
                            #         self.count_for_traffic = 0

                        #flag = 0
                        # Rewrite flag

                        self.flag_sign = 0

                        # shared = {"Object":0}
                        # traffic_detection = rospy.get_param('~traffic_detection')
                        # fp = open(traffic_detection,"w")
                        # pickle.dump(shared, fp)

                    if self.flag_sign == 1:

                        rospy.loginfo("Right traffic sign detection!")
                        time_test = time.time()
                        #while time.time() -time_test <= 0.3: 
                        while self.steering_bool:
                            #print("HEREEEE")
                            #self.subscriber1.unregister()
                            #flag = 1        steering_bool = True
                            #rospy.loginfo("This time to port to right!")
                            #angle= Float32()

                            # raw =  connect.subscribe('g2_never_die/camera/rgb/compressed', CompressedImage())

                            # frame = callback_image(raw)

                            # self.subscriber1 = rospy.Subscriber("g2_never_die/camera/rgb/compressed",
                            #                                     CompressedImage,
                            #                                     self.callback,
                            #                                     queue_size=1)

                            # self.balance_image = rospy.Subscriber("g2_never_die/camera/rgb/compressed",
                            #                                     CompressedImage,
                            #                                     self.callback,
                            #                                     queue_size=1)

                            # self.balance = rospy.Subscriber("g2_never_die/balance",
                            #                                     Int8,
                            #                                     self.balance_lane,
                            #                                     queue_size=1)  
                            
                                
                            # lane_detection = rospy.get_param('~lane_detection')

                            # fp_abc = open(lane_detection)

                            # shared_abc = pickle.load(fp_abc)

                            # lane = shared_abc["Lane"]

                            # self.balance_road = lane

                            mes = rospy.wait_for_message("g2_never_die/balance",Int8,timeout=50)

                            print(mes)
                            
                            self.balance_road = mes.data
                    
                            angle= Float32()
                            speed_car = Float32()
                            angle.data = 90
                            speed_car.data = 0
                            self.speed.publish(speed_car)
                            self.angle_car.publish(angle)
                            
                            print("BALANCE: ",self.balance_road)

                            if self.balance_road != 0:
                                rospy.logerr("Hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
                                self.count_for_traffic += 1
                                if self.count_for_traffic == 15:
                                    self.steering_bool = False
                                    self.count_for_traffic = 0

                            #rospy.spin()

                        # flag = 0
                        # Rewrite flag
                        # shared = {"Object":0}
                        # traffic_detection = rospy.get_param('~traffic_detection')
                        # fp = open(traffic_detection,"w")
                        # pickle.dump(shared, fp)

                        #self.balance_road = 0

                        self.flag_sign = 0

                    # self.subscriber1 = rospy.Subscriber("g2_never_die/camera/rgb/compressed",
                    #                                     CompressedImage,
                    #                                     self.callback,
                    #                                     queue_size=1)                         
                    self.nga_tu = 0
                    
        except Exception, e:
            print(str(e))

if __name__ == '__main__':

    rospy.init_node('Goodgame_traffictask', anonymous=True)

    traffic = traffic_task()

    while not rospy.is_shutdown():
        # define frame object and detect
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print "Shutting down ROS Image feature detector module"
    cv2.destroyAllWindows()



