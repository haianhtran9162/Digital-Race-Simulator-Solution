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
import obstacle

def callback_image(raw):
    np_arr = np.fromstring(raw.data, np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image_np

class depth_task:
    
    def __init__(self):

        # self.subscriber1 = rospy.Subscriber("g2_never_die/camera/rgb/compressed",
        #                                     CompressedImage,
        #                                     self.callback,
        #                                     queue_size=1)
        self.subscriber2 = rospy.Subscriber("g2_never_die/camera/depth/compressed",
                                            CompressedImage,
                                            self.callback_depth,
                                            queue_size=1)

        self.speed = rospy.Publisher("g2_never_die/set_speed",Float32,queue_size = 1)

        self.angle_car = rospy.Publisher("g2_never_die/set_angle",Float32,queue_size = 1)

        self.traffic_sign = rospy.Publisher("g2_never_die/traffic_sign",Int8,queue_size = 1)

        self.depth_x = rospy.Publisher("g2_never_die/depth_x",Float32,queue_size = 1)

        self.depth_y = rospy.Publisher("g2_never_die/depth_y",Float32,queue_size = 1)

    def callback(self, ros_data):

        np_app = np.fromstring(ros_data.data,np.uint8)

        frame = cv2.imdecode(np_app,cv2.IMREAD_COLOR)
        
        cv2.imshow("Frame",frame)

        cv2.waitKey(1)   

    def callback_depth(self,ros_data):

        frame_depth,depth_raw = processing_depth(ros_data)   

        gray_ex = cv2.cvtColor(frame_depth,cv2.COLOR_BGR2GRAY)

        image_np = cv2.resize(gray_ex, (320, 240))

        image_np = image_np[100:, :]
        
        obstacle_detector = obstacle.detection.DepthProcessor()

        danger_zone = obstacle_detector.combine(image_np * 10)   

        if danger_zone[0] != 0 and danger_zone != 0:

            x = Float32()
            y = Float32()
            x = danger_zone[0]
            y = danger_zone[1]

            self.depth_x.publish(x)
            self.depth_y.publish(y)

            print(x,y) 

        #cv2.imshow("Depth",frame_depth)
        
        cv2.waitKey(1)   

if __name__ == '__main__':

    rospy.init_node('Goodgame_depthtask', anonymous=True)

    depth_task = depth_task()

    while not rospy.is_shutdown():
        # define frame object and detect
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print "Shutting down ROS Image feature detector module"
    cv2.destroyAllWindows()


# if __name__ == '__main__':
    
#     rospy.init_node('Goodgame_depthtask', anonymous=True)
    
#     connect = WebsocketROSClient('127.0.0.1', 9090) # ip, port, name of client

#     connect_get_depth = WebsocketROSClient('127.0.0.1', 9090) # ip, port, name of client

#     try:
#         rospy.loginfo("Connect successfully for Depth lane! Goodluck Goodgame!")

#         value = Int8()
        
#         middle_pos_x = 160        
#         middle_pos_y = 240
        
#         count = 0
#         pre_count = 0
#         old_count = 0
#         steer_angle = 0
#         middle_pos_x_new = 0
        
#         while not rospy.is_shutdown():
            
#             # subscribe
#             #pose = Float32()
#             #pose.data = 40
#             #connect.publish('g2_never_die/set_speed',pose)

#             shared = {"obstacle":0}

#             depth_detection = rospy.get_param('~depth_detection')

#             fp = open(depth_detection,"w")
            
#             pickle.dump(shared, fp)

#             raw =  connect.subscribe('g2_never_die/camera/rgb/compressed', CompressedImage())

#             frame = callback_image(raw)

#             lane_detection = laneDetection(frame)

#             depth_raw = connect_get_depth.subscribe('g2_never_die/camera/depth/compressed', CompressedImage())

#             frame_depth,depth_raw = processing_depth(depth_raw)
            
#             old_count = count

#             try:
                                
#                 line_segments,lines,components = lane_detection.processImage()
                
#                 corr_img,gray_ex,cleaned,masked = lane_detection.processBirdView()

#                 transform_matrix,warped_image = lane_detection.perspective()

#                 steering = steering_angle(corr_img,lines,transform_matrix)

#                 lane_lines = steering.average_slop_intercept()
                
#                 angle_degree,check_lane,lane_width = steering.compute_angle_and_speed() #compute_angle_and_speed(lane_lines)

#                 angle = Float32()

#                 angle.data = -angle_degree 

#                 if len(lane_lines[0]) == 1 and len(lane_lines[1]) == 1:

#                     left_x1, left_y1, left_x2, left_y2 = lane_lines[0][0]

#                     right_x1, right_y1, right_x2, right_y2 = lane_lines[1][0]

#                     middle_pos_x = ( ( left_x1 + left_x2 ) / 2 + (right_x1 + right_x2) / 2 ) /2
#                     middle_pos_y = ( ( left_y1 + left_y2)  / 2 + (right_y1 + right_y2) /2 )  /2 
#                     #print(middle_pos)
#                 gray_ex = cv2.cvtColor(frame_depth,cv2.COLOR_BGR2GRAY)
#                 image_np = cv2.resize(gray_ex, (320, 240))
#                 image_np = image_np[80:, :]
#                 obstacle_detector = obstacle.detection.DepthProcessor()
#                 danger_zone = obstacle_detector.combine(image_np * 10)     
#                 a = danger_zone[0]
#                 b = danger_zone[1]
#                 #print(a,b)
#                 points_to_be_transformed = np.array([[[middle_pos_x,0]]], dtype=np.float32) #diem chuyen sang birdview
#                 bird_view = cv2.perspectiveTransform(points_to_be_transformed,transform_matrix['Minv'])   
                
#                 middle_pos_x = bird_view[0][0][0]
#                 #danger_zone_1_new = bird_view[0][0][1]
#                 if danger_zone != (0, 0):

#                     # 2 objects
#                     if danger_zone[0] == -1:
#                         print("2 obstacles")
#                         middle_pos_x_new = danger_zone[0]
#                     # Single object
#                     else:
#                         center_danger_zone = int((danger_zone[0] + danger_zone[1]) / 2)

#                         if danger_zone[0] < middle_pos_x < danger_zone[1]:
#                             # Obstacle's on the right
#                             if middle_pos_x < center_danger_zone:
#                                 rospy.logwarn("On the right")
#                                 #print(middle_pos_x,danger_zone[0],danger_zone[1])
#                                 middle_pos_x_new = danger_zone[0]

#                                 points_to_be_transformed = np.array([[[middle_pos_x_new,0]]], dtype=np.float32) #diem chuyen sang birdview

#                                 bird_view = cv2.perspectiveTransform(points_to_be_transformed,transform_matrix['M'])   
                                
#                                 middle_pos_x_new = bird_view[0][0][0]   

#                                 distance_x = middle_pos_x_new - 160

#                                 distance_y = middle_pos_y - 240

#                                 # Angle to middle position
#                                 steer_angle = math.atan(float(distance_x) / distance_y) * 180 / math.pi

#                                 angle.data = steer_angle

#                                 rospy.logwarn("DEPTH: " + str(steer_angle))
                                
#                                 shared = {"obstacle":steer_angle}

#                                 depth_detection = rospy.get_param('~depth_detection')

#                                 fp = open(depth_detection,"w")

#                                 pickle.dump(shared, fp)

#                                 count += 1
                                
#                             # Left
#                             else:
#                                 rospy.logwarn("On the left")
#                                 #print(middle_pos_x,danger_zone[0],danger_zone[1])
#                                 middle_pos_x_new = danger_zone[1]

#                                 points_to_be_transformed = np.array([[[middle_pos_x_new,0]]], dtype=np.float32) #diem chuyen sang birdview

#                                 bird_view = cv2.perspectiveTransform(points_to_be_transformed,transform_matrix['M'])   
                                
#                                 middle_pos_x_new = bird_view[0][0][0]   

#                                 distance_x = middle_pos_x_new - 160

#                                 distance_y = middle_pos_y - 240

#                                 # Angle to middle position
#                                 steer_angle = math.atan(float(distance_x) / distance_y) * 180 / math.pi

#                                 angle.data = steer_angle

#                                 rospy.logwarn("DEPTH: " + str(steer_angle))
                                
#                                 shared = {"obstacle":steer_angle}

#                                 depth_detection = rospy.get_param('~depth_detection')

#                                 fp = open(depth_detection,"w")

#                                 pickle.dump(shared, fp)

#                                 count += 1

#                             #####################################################


#                             #connect.publish('g2_never_die/set_angle',angle)
                
#                 if old_count != 0 and old_count == count:

#                     angle.data = steer_angle
#                     #connect.publish('g2_never_die/set_angle',angle)

#                     pre_count += 1

#                     shared = {"obstacle":steer_angle}

#                     depth_detection = rospy.get_param('~depth_detection')

#                     fp = open(depth_detection,"w")
                    
#                     pickle.dump(shared, fp)

#                     if pre_count == 10:

#                         # connect.publish('g2_never_die/set_angle',angle)
#                         #rospy.loginfo("Stop!")

#                         pre_count = 0
#                         old_count = 0
#                         count = 0     
#                         flag_car = False

#                 #print("Final",angle.data)
#                 #connect.publish('g2_never_die/set_angle',angle)
                
#                 #cv2.imshow("Gray",image_np)

#             except Exception, e:
#                 print(str(e))
#             cv2.waitKey(1)


#     except KeyboardInterrupt:
#         pass

#     rospy.spin()
