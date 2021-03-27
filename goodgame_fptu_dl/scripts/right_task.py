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
import time
import pickle

def callback_image(raw):
    np_arr = np.fromstring(raw.data, np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image_np

class right_sign:
    
    def __init__(self):

        self.subscriber1 = rospy.Subscriber("g2_never_die/camera/rgb/compressed",
                                            CompressedImage,
                                            self.callback,
                                            queue_size=1)

        self.speed = rospy.Publisher("g2_never_die/set_speed",Float32,queue_size = 1)

        self.angle_car = rospy.Publisher("g2_never_die/set_angle",Float32,queue_size = 1)

        self.traffic_sign = rospy.Publisher("g2_never_die/traffic_sign",Int8,queue_size = 1)

        right_src = rospy.get_param('~right_traffic_sign') #'./object_detection/cascade_right_sign.xml'

        self.traffic_right = cv2.CascadeClassifier(right_src)

    def callback(self, ros_data):

        np_app = np.fromstring(ros_data.data,np.uint8)

        frame = cv2.imdecode(np_app,cv2.IMREAD_COLOR)
        
        gray_ex = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        right =  self.traffic_right.detectMultiScale(gray_ex,1.1,3)

        for (x,y,w,h) in right:
            #print("detection")

            #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            
            msg = Int8()

            msg.data = 1

            self.traffic_sign.publish(msg)
            
            #connect.publish('g2_never_die/traffic_value',traffic_value)

            #cv2.imwrite("/home/huyphan/Desktop/right.jpg",frame)

        #cv2.imshow("right",frame)  

        cv2.waitKey(1)     

if __name__ == '__main__':

    rospy.init_node('Goodgame_righttask', anonymous=True)

    right = right_sign()

    while not rospy.is_shutdown():
        # define frame object and detect
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print "Shutting down ROS Image feature detector module"
    cv2.destroyAllWindows()

# if __name__ == '__main__':
    
#     rospy.init_node('Goodgame_righttask', anonymous=True)

#     connect = WebsocketROSClient('127.0.0.1', 9090) # ip, port, name of client
    
#     try:
#         rospy.loginfo("Connect successfully for right sign! Goodluck Goodgame!")

#         right_src = rospy.get_param('~right_traffic_sign') #'./object_detection/cascade_right_sign.xml'

#         traffic_right = cv2.CascadeClassifier(right_src)

#         traffic_value = Int8()

#         traffic_value.data = 1
        
#         while not rospy.is_shutdown():

#             raw =  connect.subscribe('g2_never_die/camera/rgb/compressed', CompressedImage())
            
#             frame = callback_image(raw)

#             gray_ex = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

#             right =  traffic_right.detectMultiScale(gray_ex,1.1,3)


#             for (x,y,w,h) in right:
#                 #print("detection")

#                 cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

#                 #connect.publish('g2_never_die/traffic_value',traffic_value)

#                 traffic_detection = rospy.get_param('~traffic_detection')

#                 shared = {"Object":1}

#                 fp = open(traffic_detection,"w")

#                 pickle.dump(shared, fp)  

#                 #cv2.imwrite("/home/huyphan/Desktop/right.jpg",frame)

#             #cv2.imshow("right",frame)  

#             cv2.waitKey(1)


#     except KeyboardInterrupt:
#         pass
#     rospy.spin()
