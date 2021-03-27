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

if __name__ == '__main__':
    
    rospy.init_node('Goodgame_balancetask', anonymous=True)

    connect = WebsocketROSClient('127.0.0.1', 9090) # ip, port, name of client
    
    try:
        rospy.loginfo("Connect successfully for balance lane! Goodluck Goodgame!")

        value = Int8()

        while not rospy.is_shutdown():
            
            # subscribe

            raw =  connect.subscribe('g2_never_die/camera/rgb/compressed', CompressedImage())

            frame = callback_image(raw)

            lane_detection = laneDetection(frame)

            try:
                line_segments,lines,components = lane_detection.processImage()
                
                corr_img,gray_ex,cleaned,masked = lane_detection.processBirdView()

                transform_matrix,warped_image = lane_detection.perspective()

                steering = steering_angle(corr_img,lines,transform_matrix)

                lane_lines = steering.average_slop_intercept()

                angle_degree,check_lane,lane_width = steering.compute_angle_and_speed()

                #print("ANGLEEEEEEEEEEEEEEEEEEEEEEEE: ",angle_degree,check_lane,lane_width)

                if not check_lane is None and check_lane !=0 and abs(angle_degree) > 30:
                    #print("Result: ",check_lane)

                    shared = {"Lane":1}
                    
                    lane_choose = rospy.get_param('~lane_detection')

                    fp = open(lane_choose,"w")

                    pickle.dump(shared, fp)

                if check_lane == 0 and lane_width > 60 and abs(angle_degree) > 10:

                    shared = {"Lane":2}
                    
                    lane_choose = rospy.get_param('~lane_detection')

                    fp = open(lane_choose,"w")

                    pickle.dump(shared, fp)

            except Exception, e:

                shared = {"Lane":0}

                lane_choose = rospy.get_param('~lane_detection')

                fp = open(lane_choose,"w")

                pickle.dump(shared, fp)
            

            #cv2.imshow("Balance",frame)         
            
            cv2.waitKey(1)


    except KeyboardInterrupt:
        pass

    rospy.spin()
