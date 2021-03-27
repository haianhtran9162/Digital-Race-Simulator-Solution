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

from utils import *
from rosws import *
import math
from lane_detection import laneDetection
from steering_angle_processing import steering_angle
from traffic_sign_detection import traffic_sign
from numpy import linalg as LA
from depth_processing import *
import pickle
import time

def callback_image(raw):
    np_arr = np.fromstring(raw.data, np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image_np

if __name__ == '__main__':
    
    rospy.init_node('main_cv2imshow_left', anonymous=True)

    connect = WebsocketROSClient('127.0.0.1', 9090) # ip, port, name of client
    
    #connect_get_depth = WebsocketROSClient('127.0.0.1', 9090) # ip, port, name of client

    try:
        rospy.loginfo("Connect successfully for left sign! Goodluck Goodgame!")
        
        count = 0

        while not rospy.is_shutdown():
            
            # subscribe
            raw =  connect.subscribe('g2_never_die/camera/rgb/compressed', CompressedImage())
            
            frame = callback_image(raw)

            filename ="/home/dylan/Desktop/Data_Aug/simulator3_" + str(count) +".png"
            
            cv2.imwrite(filename,frame)

            count += 1
            
    except KeyboardInterrupt:
        pass
    rospy.spin()
