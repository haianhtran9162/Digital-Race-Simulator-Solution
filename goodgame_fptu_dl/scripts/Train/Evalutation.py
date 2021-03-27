import glob
import random
import time

import cv2

from keras.applications import MobileNetV2
from keras.callbacks import Callback
import sys
from keras.models import Model
from keras.layers import Dense, Flatten, Input, Concatenate, TimeDistributed, LSTM, merge, Conv2D, Permute, \
    Conv2DTranspose, Reshape, Activation, AveragePooling2D, Conv2D, Dropout, MaxPooling2D, UpSampling2D, Cropping2D
from keras.optimizers import Adam
import numpy as np
import common
import keras.backend as K
import image_processing as imgproc
from keras.layers.normalization import BatchNormalization

from read_json import read_json
from rosws import *

image_paths = glob.glob('/content/test_unet/*.png')


def Pun_net():
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

def callback(raw):
    np_app = np.fromstring(raw.data,np.uint8)
    frame = cv2.imdecode(np_app,cv2.IMREAD_COLOR)
    return frame
    
# while True:
#     x,y= next(generator(1))
#     image = x[0]
#     seg = y[0]
#     cv2.imshow('test',np.concatenate((image,seg),1))
#     cv2.waitKey(0)



#model = Pun_net()
#model.summary()
#model.compile(loss='binary_crossentropy', optimizer='rmsprop',metrics=['binary_accuracy'])
#model.load_weights('/content/model.h5')
#callback = My_Callback()
#model.fit_generator(generator_json(32), steps_per_epoch=500, epochs=25, callbacks=[callback])
#model.save_weights('./pun_net_backup.h5')


if __name__ == '__main__':
    
    rospy.init_node('Goodgame_websocket', anonymous=True)

    connect = WebsocketROSClient('127.0.0.1', 9090) # ip, port, name of client

    try:
        rospy.loginfo('Connected ROS succesfully!')
        model = Pun_net()
        model.summary()
        model.compile(loss='binary_crossentropy', optimizer='rmsprop',metrics=['binary_accuracy'])
        model.load_weights('./model.h5')

        while not rospy.is_shutdown():

            # Publish to server
            begin_time=time.time()
            pose = Float32()
            pose.data = 40
            ##print("Publishing to server")

            connect.publish('team1/set_speed', pose)
            
            raw = connect.subscribe('team1/camera/rgb/compressed',CompressedImage())

            frame = callback(raw)
            cv2.imshow("Frame",frame)
            cv2.waitKey(1)