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


def callback_image(raw):
    np_arr = np.fromstring(raw.data, np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image_np

class balance_task:
    
    def __init__(self):

        self.balance_image = rospy.Subscriber("g2_never_die/camera/rgb/compressed",
                                            CompressedImage,
                                            self.callback,
                                            queue_size=1)

        self.balance = rospy.Publisher("g2_never_die/position",Int8,queue_size = 1)

        self.speed = rospy.Publisher("g2_never_die/set_speed",Float32,queue_size = 1)

        self.angle_car = rospy.Publisher("g2_never_die/set_angle",Float32,queue_size = 1)

        self.flag_sign = 0

        self.nga_tu = 0

        self.flag = 0 

        self.steering_bool = False

        self.count_for_traffic = 0

        self.balance_road = 0

        self.coor = rospy.Subscriber("g2_never_die/car_coor", Float32, self.callback_coor)

        self.car_x = 0

        self.car_y = 0

        self.i = 0

    def callback_coor(self,data):

        #print(data.data)
        self.car_x = (data.data)[0]
        self.car_y = (data.data)[1]

    def callback(self, ros_data):

        np_app = np.fromstring(ros_data.data,np.uint8)

        self.frame = cv2.imdecode(np_app,cv2.IMREAD_COLOR)
        
        if self.i%5 == 0:
            anh = '/home/dylan/Desktop/Data_Final_CMM/Data_FN_CMM_2_' + str(self.i) + '.png'
            cv2.imwrite(anh,self.frame)
        self.i = self.i + 1
        
        cv2.waitKey(1)   


if __name__ == '__main__':

    rospy.init_node('Goodgame_balancetask', anonymous=True)

    balance = balance_task()

    while not rospy.is_shutdown():
        # define frame object and detect
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print "Shutting down ROS Image feature detector module"
    cv2.destroyAllWindows()



