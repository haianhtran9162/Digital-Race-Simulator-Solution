#!/usr/bin/env python

import glob
import random
import time
import numpy as np
import cv2
import math 
import pickle
from numpy import linalg as LA

import tensorflow as tf

from keras.applications import MobileNetV2
from keras.callbacks import Callback
import sys
from keras.models import Model
from keras.layers import Dense, Flatten, Input, Concatenate, TimeDistributed, LSTM, merge, Conv2D, Permute, \
   Conv2DTranspose, Reshape, Activation, AveragePooling2D, Conv2D, Dropout, MaxPooling2D, UpSampling2D, Cropping2D
from keras.optimizers import Adam

import keras.backend as K
from keras.layers.normalization import BatchNormalization

import algorithms

from algorithms.rosws import *
from algorithms.utils import *

from algorithms.lane_detection import laneDetection
from algorithms.steering_angle_processing import steering_angle
from algorithms.traffic_sign_detection import traffic_sign
from depth_processing import *

import obstacle

### TEST FOR SSD ###
from keras.optimizers import Adam
from keras import backend as K
from models.keras_ssd7 import build_model
from keras_loss_function.keras_ssd_loss import SSDLoss
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

import cv2
import time
import glob
import json
import os
import numpy as np
import tensorflow as tf

######################

def G2_Unet():
    image_input = Input((1, 84, 84))
    if K.common.image_dim_ordering() == 'tf':
        image_p = Permute((2, 3, 1))(image_input)
    elif K.common.image_dim_ordering() == 'th':
        image_p = Permute((1, 2, 3))(image_input)
    conv1 = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(image_p)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(conv1)
    print(conv1.shape)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), strides=1, activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), strides=1, activation='relu', padding='same')(conv3)

    up1 = Concatenate()([UpSampling2D(size=(2, 2))(conv3), conv2])
    conv4 = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(conv4)

    up2 = Concatenate()([UpSampling2D(size=(2, 2))(conv4), conv1])
    conv5 = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(conv5)

    conv6 = Conv2D(1, 1, strides=1, activation='sigmoid', padding='same')(conv5)
    outputs = Permute((3, 1, 2))(conv6)
    model = Model(input=image_input, output=outputs)

    return model

class deep_car:
    
    global flag

    def __init__(self):
        
        K.clear_session() # Clear previous models from memory.

        self.nga_tu = 0

        self.flag = 0 

        self.steering_bool = False

        self.count_for_traffic = 0

        self.count=0

        self.pre_count=0

        self.old_count=0
        
        self.flag_car = False

        self.position_count_left = 0

        self.position_count_right = 0

        self.vi_tri_vat_can = 0

        self.vi_tri_vat_can_hough = 0
        
        config = tf.ConfigProto()

        config.gpu_options.allow_growth = True

        self.session = tf.InteractiveSession(config=config)
        
        self.graph = tf.get_default_graph()

        self.model = G2_Unet()

        self.model.summary()

        model_path = rospy.get_param('~model_dl')

        self.model.load_weights(model_path)

        self.subscriber1 = rospy.Subscriber("g2_never_die/camera/rgb/compressed",
                                            CompressedImage,
                                            self.callback,
                                            queue_size=10)

        self.speed = rospy.Publisher("g2_never_die/set_speed",Float32,queue_size = 1)

        self.angle_car = rospy.Publisher("g2_never_die/set_angle",Float32,queue_size = 1)

        # self.pub = rospy.Publisher('g2_never_die/car_coor', numpy_msg(Floats),queue_size=10)

        # self.traffic_sign = rospy.Subscriber("g2_never_die/traffic_sign",
        #                                     Int8,
        #                                     self.callback_sign,
        #                                     queue_size=1)

        # self.balance = rospy.Subscriber("g2_never_die/position",
        #                                    Int8,
        #                                    self.position,
        #                                   queue_size=1) 

        # self.depth_x = rospy.Subscriber("g2_never_die/depth_x",Float32,self.depth_x_callback,queue_size=1)

        # self.depth_y = rospy.Subscriber("g2_never_die/depth_y",Float32,self.depth_y_callback,queue_size=1)
                                                 
        self.flag_sign = 0

        self.balance_road = 0

        self.den_diem_re = False

        self.stop_here = False

        self.depth_x_coor = 0

        self.depth_y_coor = 0

        self.check_point = 0

        self.count_for_car = 0

        self.check_car = False

        self.last_angle = 0

        self.check_balance = False
        
        self.flag_for_re = False

        ## SSD ##

        img_height = 240 # Height of the input images
        img_width = 320 # Width of the input images
        img_channels = 3 # Number of color channels of the input images
        intensity_mean = 127.5 # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
        intensity_range = 127.5 # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
        n_classes = 3 # Number of positive classes
        scales = [0.08, 0.16, 0.32, 0.64, 0.96] # An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.
        aspect_ratios = [0.5, 1.0, 2.0] # The list of aspect ratios for the anchor boxes
        two_boxes_for_ar1 = True # Whether or not you want to generate two anchor boxes for aspect ratio 1
        steps = None # In case you'd like to set the step sizes for the anchor box grids manually; not recommended
        offsets = None # In case you'd like to set the offsets for the anchor box grids manually; not recommended
        clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
        variances = [1.0, 1.0, 1.0, 1.0] # The list of variances by which the encoded target coordinates are scaled
        self.normalize_coords = True # Whether or not the model is supposed to use coordinates relative to the image size
        # weight_path = 'trained_models/digit_detect_pretrained.h5'
        weight_path =  rospy.get_param('~ssd_model')

        self.ssd_model = build_model(image_size=(img_height, img_width, img_channels),
                    n_classes=n_classes,
                    mode='training',
                    l2_regularization=0.0005,
                    scales=scales,
                    aspect_ratios_global=aspect_ratios,
                    aspect_ratios_per_layer=None,
                    two_boxes_for_ar1=two_boxes_for_ar1,
                    steps=steps,
                    offsets=offsets,
                    clip_boxes=clip_boxes,
                    variances=variances,
                    normalize_coords=self.normalize_coords,
                    subtract_mean=intensity_mean,
                    divide_by_stddev=intensity_range)

        if(weight_path is not None):
            self.ssd_model.load_weights(weight_path, by_name=True)

        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

        self.ssd_model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

    def position(self,data):

        #print(data.data)
        self.vi_tri_vat_can_hough = data.data

    def get_ypred_decoded(self,r_img):
        '''
            Perform prediction on 1 image
            
            Arguments:
                r_img: rgb images which is reshaped to (1, h, w, c)
            Returns: Raw prediction that needed to be decode -> bboxes and labels
        '''
        y_pred = self.ssd_model.predict(r_img)
        #y_pred = model.predict(r_img)
        y_pred_decoded = decode_detections(y_pred,
                                        confidence_thresh=0.61,
                                        iou_threshold=0.1,
                                        top_k=200,
                                        normalize_coords=self.normalize_coords,
                                        img_height=240,
                                        img_width=320)
        
        return y_pred_decoded[0]

    def callback(self, ros_data):
        
        with self.session.as_default():
            with self.graph.as_default():
            #### direct conversion to CV2 ####
                np_app = np.fromstring(ros_data.data,np.uint8)

                self.frame = cv2.imdecode(np_app,cv2.IMREAD_COLOR)
                cv2.imshow("hello",self.frame)
                #rospy.logwarn("FLAGG " + str(self.den_diem_re))
                try: 
                    # Publish to server
                    if self.den_diem_re == False:
                        
                    #self.flag_car = True

                        pose = Float32()

                        pose.data = 50

                        self.speed.publish(pose)
                        print("hello")
                        
                        self.old_count = self.count

                        self.traffic_detection_sign()   

                        if self.flag_car == False:
                                                
                            self.frame_normal = self.frame[40:, :].copy()

                            resized = cv2.resize(self.frame_normal, (84, 84))

                            gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY) / 255.0
                            # i_d = detector.detect_object(mat)
                            # i_d.visualize()
                            #with self.session.as_default():
                            #    with self.graph.as_default():
                            y = self.model.predict(np.array([[gray]]))
                            gray *= 255.
                            y *= 255
                            # print(image.shape,y[0][0].shape)
                            #cv2.imshow("Deep learning",y[0][0])
                            self.y_true = cv2.resize(y[0][0],(320,240))
                            self.src_pts = np.float32([[0,85],[320,85],[320,240],[0,240]])
                            self.dst_pts = np.float32([[0,0],[320,0],[200,240],[120,240]])
                            # src_pts = np.float32([[0,0],[320,0],[215,240],[105,240]])
                            # dst_pts = np.float32([[0,0],[240,0],[240,320],[0,320]])
                            self.transform_matrix = perspective_transform(self.src_pts,self.dst_pts) #From utils.py
                            
                            warped_image = birdView(self.y_true,self.transform_matrix['M'])
                            #cv2.imshow("Bird",warped_image)

                            _, contours,  _ = cv2.findContours(warped_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                            contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]

                            biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

                            mask_after = np.zeros(self.frame.shape, np.uint8)

                            #cv2.drawContours(mask, [biggest_contour], -1, 255, -1)

                            #cv2.imshow("demo",mask)

                            if not len(biggest_contour) == 0:

                                cv2.drawContours(mask_after, [biggest_contour], -1, (255), -1) #Approx
                                ############ TEST #######################
                                extLeft = tuple(biggest_contour[biggest_contour[:, :, 0].argmin()][0])
                                extRight = tuple(biggest_contour[biggest_contour[:, :, 0].argmax()][0])
                                #extTop = tuple(biggest_contour[biggest_contour[:, :, 1].argmin()][0])
                                #extBot = tuple(biggest_contour[biggest_contour[:, :, 1].argmax()][0])
                                cv2.circle(mask_after, extLeft, 8, (0, 0, 255), -1)
                                cv2.circle(mask_after, extRight, 8, (0, 255, 0), -1)
                                #cv2.circle(mask_after, extTop, 8, (255, 0, 0), -1)
                                #cv2.circle(mask_after, extBot, 8, (255, 255, 0), -1)
                                #########################################
                                M = cv2.moments(biggest_contour)
                            
                                # calculate x,y coordinate of center
                                if not M["m00"] == 0:

                                    cX_center = int(M["m10"] / M["m00"])

                                    cY_center = int(M["m01"] / M["m00"])

                                    cv2.circle(mask_after, (cX_center, cY_center), 5, (102, 42, 255), -1)

                                    cv2.putText(mask_after, "centroid", (cX_center - 25, cY_center - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (102, 42, 255), 2)   
                                    
                                    slope = (cX_center - 160) / float(cY_center - 240)

                                    angle_radian= float(math.atan(slope))

                                    angle_degree= float(angle_radian * 180.0 / math.pi)

                                    #self.last_angle = angle_degree
                                    #rospy.logwarn("GOC NORMAL: "  + str(angle_degree))
                                    angle = Float32()

                                    angle.data= -angle_degree

                                    #while time.time() -start <= 0.02:
                                    #connect.publish('g2_never_die/set_angle',angle)
                                    self.angle_car.publish(angle)

                            #cv2.imshow("Normal",mask_after)


                        # with self.session.as_default():
                        #     with self.graph.as_default():
                                
                        image = cv2.resize(self.frame, (320, 240))

                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                        image = np.reshape(image, (1, 240, 320, 3))

                        y_pred_decoded = self.get_ypred_decoded(image)

                        if y_pred_decoded != []:                

                            #self.frame = cv2.rectangle(self.frame, (int(y_pred_decoded[0][2]),int(y_pred_decoded[0][3])), (int(y_pred_decoded[0][4]),int(y_pred_decoded[0][5])), (0,0,255), 2)
                            centroid_x = int(y_pred_decoded[0][2]) + ( ( int(y_pred_decoded[0][4]) -  int(y_pred_decoded[0][2]) ) / 2 )
                            centroid_y = int(y_pred_decoded[0][3]) + ( ( int(y_pred_decoded[0][5]) -  int(y_pred_decoded[0][3]) ) / 2 )            
                            #cv2.circle(self.frame, (centroid_x, centroid_y), 5, (102, 42, 255), -1)

                            if y_pred_decoded[0][0] == 1 and y_pred_decoded[0][1] > 0.9:
                                ##CAR  HERE ###

                                self.count_for_car += 1

                                # car_coors = np.array([centroid_x,centroid_y], dtype=np.float32)

                                # self.pub.publish(car_coors) # Gui toa do xe

                                self.flag_car = True
                                self.count+=1

                                #cv2.putText(self.frame, "Car", (centroid_x - 25, centroid_y - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (102, 42, 255), 2)  
                                                
                                points_to_be_transformed = np.array([[[centroid_x,centroid_y]]], dtype=np.float32) #diem chuyen sang birdview
                                bird_view = cv2.perspectiveTransform(points_to_be_transformed,self.transform_matrix['M'])                

                                x_bird = bird_view[0][0][0]
                                y_bird = bird_view[0][0][1]

                                self.frame_for_lane = self.frame[40:, :].copy()

                                resized = cv2.resize(self.frame_for_lane, (84, 84))

                                gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY) / 255.0
                                # i_d = detector.detect_object(mat)
                                # i_d.visualize()
                                # with self.session.as_default():
                                #     with self.graph.as_default():
                                y = self.model.predict(np.array([[gray]]))
                        
                                gray *= 255.
                                y *= 255
                                # print(image.shape,y[0][0].shape)

                                self.y_true = cv2.resize(y[0][0],(320,240))

                                _,contours_goc, _ = cv2.findContours(self.y_true.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                                contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours_goc]

                                biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

                                left = tuple(biggest_contour[biggest_contour[:, :, 0].argmin()][0])

                                right = tuple(biggest_contour[biggest_contour[:, :, 0].argmax()][0])

                                i = 0

                                for contour in biggest_contour:
                                    for (x,y) in contour:
                                        if y > centroid_y - 10:
                                            biggest_contour = np.delete(biggest_contour,i,axis=0)
                                            i = i - 1
                                    i = i + 1


                                M = cv2.moments(biggest_contour)
                            
                                # calculate x,y coordinate of center
                                if not M["m00"] == 0:

                                    cX_center_goc = int(M["m10"] / M["m00"])

                                    cY_center_goc = int(M["m01"] / M["m00"])

                                    #mask_position = np.zeros(self.frame.shape, np.uint8)
                                    
                                    #cv2.drawContours(mask_position, [biggest_contour], -1, (255), -1) #Approx
                                    
                                    #cv2.circle(mask_position, (cX_center_goc, cY_center_goc), 5, (102, 42, 255), -1)

                                    #cv2.circle(mask_position, (centroid_x, centroid_y), 5, (0, 255, 0), -1)

                                    #cv2.imshow("Goc",mask_test)
                                    #print("VI TRI: ", left[0],centroid_x,right[0])

                                    if cX_center_goc > centroid_x and left[0] < centroid_x < right[0]:
                                        self.vi_tri_vat_can = -1
                                        rospy.logwarn("Car is on the left lane!")
                                        
                                    elif cX_center_goc < centroid_x and left[0] < centroid_x < right[0]:
                                        rospy.logwarn("Car is on the right lane!")
                                        self.vi_tri_vat_can = 1


                                self.src_pts = np.float32([[0,85],[320,85],[320,240],[0,240]])
                                self.dst_pts = np.float32([[0,0],[320,0],[200,240],[120,240]])
                                # src_pts = np.float32([[0,0],[320,0],[215,240],[105,240]])
                                # dst_pts = np.float32([[0,0],[240,0],[240,320],[0,320]])
                                self.transform_matrix = perspective_transform(self.src_pts,self.dst_pts) #From utils.py

                                warped_image = birdView(self.y_true,self.transform_matrix['M'])
                                
                                #cv2.imshow("Bird",warped_image)

                                _, contours, _ = cv2.findContours(warped_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                                contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]

                                biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

                                #print(left,right)

                                #mask_after = np.zeros(self.frame.shape, np.uint8)

                                # mask_zero = np.zeros(self.frame.shape, np.uint8)

                                #mask_test = np.zeros(self.frame.shape, np.uint8)
                                
                                #cv2.drawContours(mask_after, [biggest_contour], -1, (255), -1) #Approx

                                #cv2.imshow("AFTER",mask_after)

                                M_before = cv2.moments(biggest_contour)

                                cX_center_before = int(M_before["m10"] / M_before["m00"])

                                if self.vi_tri_vat_can == -1:
                                    #print("Here")
                                    cX_center_before = cX_center_before + 12.5
                                elif self.vi_tri_vat_can == 1:
                                    #print("Here2")
                                    cX_center_before = cX_center_before - 12.5

                                cY_center_before = int(M_before["m01"] / M_before["m00"])
                                
                                slope = (cX_center_before - 160) / float(cY_center_before - 240)

                                angle_radian= float(math.atan(slope))

                                angle_degree_goc= float(angle_radian * 180.0 / math.pi)

                                rospy.logwarn("MAXIMUM: " + str(angle_degree_goc))

                                angle = Float32()

                                angle.data = -angle_degree_goc

                                self.angle_car.publish(angle)

                            elif y_pred_decoded[0][0] == 2 and y_pred_decoded[0][1] > 0.98:
                                self.flag_sign = -1
                                cv2.putText(self.frame, "Left", (centroid_x - 25, centroid_y - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (102, 42, 255), 2)   
                            elif y_pred_decoded[0][0] == 3 and y_pred_decoded[0][1] > 0.98:
                                self.flag_sign = 1
                                cv2.putText(self.frame, "Right", (centroid_x - 25, centroid_y - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (102, 42, 255), 2)   

                        
                        #cv2.imshow("frame2", self.frame)
        
                        gray_ex = cv2.cvtColor(self.frame,cv2.COLOR_BGR2GRAY)

                        if self.old_count != 0 and self.old_count == self.count:
                            # with self.session.as_default():
                            #     with self.graph.as_default():

                            self.transform_matrix = perspective_transform(self.src_pts,self.dst_pts) #From utils.py

                            warped_image = birdView(self.y_true,self.transform_matrix['M'])
                            
                            #cv2.imshow("Bird",warped_image)

                            _, contours, _ = cv2.findContours(warped_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                            contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]

                            biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

                            #left = tuple(biggest_contour[biggest_contour[:, :, 0].argmin()][0])

                            #right = tuple(biggest_contour[biggest_contour[:, :, 0].argmax()][0])

                            #print(left,right)
                            #mask_after = np.zeros(self.frame.shape, np.uint8)

                            # mask_zero = np.zeros(self.frame.shape, np.uint8)

                            #mask_test = np.zeros(self.frame.shape, np.uint8)
                            
                            #cv2.drawContours(mask_after, [biggest_contour], -1, (255), -1) #Approx

                            #cv2.imshow("AFTER",mask_after)

                            M_before = cv2.moments(biggest_contour)

                            #extLeft = tuple(biggest_contour[biggest_contour[:, :, 0].argmin()][0])

                            #extRight = tuple(biggest_contour[biggest_contour[:, :, 0].argmax()][0])

                            cX_center_before = int(M_before["m10"] / M_before["m00"])

                            if self.vi_tri_vat_can == -1:
                                cX_center_before = cX_center_before + 12.5
                            elif self.vi_tri_vat_can == 1:
                                cX_center_before = cX_center_before - 12.5

                            cY_center_before = int(M_before["m01"] / M_before["m00"])
                            
                            slope = (cX_center_before - 160) / float(cY_center_before - 240)

                            angle_radian= float(math.atan(slope))

                            angle_degree_goc= float(angle_radian * 180.0 / math.pi)

                            #rospy.logwarn("MAXIMUM: " + str(angle_degree_goc))

                            angle = Float32()

                            angle.data = -angle_degree_goc

                            self.angle_car.publish(angle)

                            self.pre_count += 1
                            
                            print(self.count_for_car)

                            if self.pre_count == 1:
                                
                                self.count_for_car = 0

                                rospy.loginfo("Stop!")

                                self.pre_count = 0

                                self.old_count = 0

                                self.count = 0     

                                self.flag_car = False

                                self.vi_tri_vat_can = 0  

                    else:

                        #cv2.imshow("ELSE",self.frame)
                        
                        lane_detection = laneDetection(self.frame)

                        if self.flag_sign == 1:

                            rospy.loginfo("Right traffic sign detection!")
                    
                            angle= Float32()
                            speed_car = Float32()
                            angle.data = 90
                            speed_car.data = 0
                            self.speed.publish(speed_car)
                            self.angle_car.publish(angle)

                        if self.flag_sign == -1:

                            rospy.loginfo("Left traffic sign detection!")
                    
                            angle= Float32()
                            speed_car = Float32()
                            angle.data = -90
                            speed_car.data = 0
                            self.speed.publish(speed_car)
                            self.angle_car.publish(angle)

                        if self.stop_here:
                            self.count_for_traffic += 1
                            if self.count_for_traffic == 2:
                                self.nga_tu = 0
                                self.steering_bool = False
                                self.count_for_traffic = 0
                                self.den_diem_re = False 
                                self.flag_sign = 0
                                self.stop_here = False
                                self.check_point += 1
                        try:

                            line_segments,lines,components = lane_detection.processImage()  
                            
                            corr_img,gray_ex,cleaned,masked = lane_detection.processBirdView()

                            transform_matrix,warped_image = lane_detection.perspective()

                            steering = steering_angle(corr_img,lines,transform_matrix)

                            lane_lines = steering.average_slop_intercept()

                            frame = steering.display()

                            angle_degree,check_lane,lane_width = steering.compute_angle_and_speed() 

                            #rospy.logwarn("GOC " + str(angle_degree))

                            if abs(angle_degree) > 20:
                                self.flag_for_re = True

                            if self.flag_for_re == True:
                                if -10 < angle_degree < 10:
                                    self.stop_here = True
                                    self.flag_for_re = False

                            # if not check_lane is None and check_lane !=0 and abs(angle_degree) > 35:

                            #     self.stop_here = True 

                            # if check_lane == 0 and abs(angle_degree) > 30:
                                
                            #     self.stop_here = True 

                        except:
                            pass

                except Exception, e:
                    rospy.logwarn(str(e))

        cv2.waitKey(1)

    def traffic_detection_sign(self):
        with self.session.as_default():
            with self.graph.as_default():
                self.frame_sign = self.frame[40:, :].copy()

                resized = cv2.resize(self.frame_sign, (84, 84))

                gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY) / 255.0
                # i_d = detector.detect_object(mat)
                # i_d.visualize()

                y = self.model.predict(np.array([[gray]]))
        #y = model.predict(np.array([[gray]]))
                gray *= 255.
                y *= 255
                # print(image.shape,y[0][0].shape)
                #cv2.imshow("Test",y[0][0])
                self.y_true = cv2.resize(y[0][0],(320,240))
                self.src_pts = np.float32([[0,85],[320,85],[320,240],[0,240]])
                self.dst_pts = np.float32([[0,0],[320,0],[200,240],[120,240]])
                # src_pts = np.float32([[0,0],[320,0],[215,240],[105,240]])
                # dst_pts = np.float32([[0,0],[240,0],[240,320],[0,320]])
                self.transform_matrix = perspective_transform(self.src_pts,self.dst_pts) #From utils.py

                warped_image = birdView(self.y_true,self.transform_matrix['M'])
                #cv2.imshow("Bird",warped_image)

                _, contours, _ = cv2.findContours(warped_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]

                # lane_detection = laneDetection(self.frame)

                # warp_img,gray_bird = lane_detection.processing_traffic_sign()         

                # _, _, contours,  _ = cv2.findContours(warp_img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]

                biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

                mask = np.zeros(self.frame.shape, np.uint8)

                cv2.drawContours(mask, [biggest_contour], -1, 255, -1)

        #cv2.imshow("Sign",mask)

                diem_re = 0

                try:
                    #print("GLAFFFFFFFFFFFFFFF",self.flag_sign)
                    if self.flag_sign == -1:
                        self.steering_bool = True
                        for i in range(0,160):
                            if np.sum(mask[50][i]) == 255:
                                diem_re += 1
                    elif self.flag_sign == 1:
                        self.steering_bool = True
                        for i in range(160,320):
                            if np.sum(mask[50][i]) == 255:
                                diem_re += 1 

                    #rospy.logwarn("Diem re: " + str(diem_re))
                    
                    if diem_re ==135:
                        self.nga_tu = 1
                        speed_car = Float32()
                        speed_car = 0
                        self.speed.publish(speed_car)

                    if self.nga_tu == 1:
                        if diem_re < 120:
                            self.den_diem_re = True
                            
                except Exception, e:
                    print(str(e))

if __name__ == '__main__':

    rospy.init_node('Goodgame_main_test', anonymous=True)
    # print("tra du=inh")
    deep = deep_car()
    
    #while not rospy.is_shutdown():
        # define frame object and detect
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down ROS Image feature detector module"

    cv2.destroyAllWindows()
