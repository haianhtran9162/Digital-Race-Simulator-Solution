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

image_paths = glob.glob('/content/test_unet/*.png')


def detectRoadwayByHLSImage(hls_image):
    _, _, s_channel = cv2.split(hls_image)

    s_channel_histogram = imgproc.getHistogram(s_channel)
    s_channel_fft_histogram = imgproc.getFFTHistogram(s_channel_histogram)

    local_minimums = imgproc.localMinimumOnFFTHistogram(s_channel_fft_histogram)

    return imgproc.inRange1Channel(s_channel, 0, local_minimums[0])


def process_image(image):
    # convert cac kenh mau khac nhau
    bgr_image = image
    gray_image = imgproc.convertBGRtoGray(bgr_image)
    hls_image = imgproc.convertBGRtoHLS(bgr_image)

    # tinh histogram, fft histogram va cac cuc tieu cua fft histogram
    gray_histogram = imgproc.getHistogram(gray_image)
    gray_fft_histogram = imgproc.getFFTHistogram(gray_histogram)

    # lay long duong bang anh HLS
    roadway_binary_image = detectRoadwayByHLSImage(hls_image)
    roadway_gray_image = cv2.bitwise_and(gray_image, gray_image, mask=roadway_binary_image)
    if (np.any(roadway_gray_image > 0)):
        roadway_gray_color_avg = int(roadway_gray_image[roadway_gray_image > 0].mean())
    else:
        roadway_gray_color_avg = 0

    # lay ra cac vung mau co dien tich lon, tim vung co dien tich lon nhat
    large_area_color = imgproc.getLargeAreaColorStrips(gray_histogram, gray_fft_histogram)
    largest_area_color = large_area_color[large_area_color[:, 4].argmax()]
    if (roadway_gray_color_avg >= largest_area_color[0] and roadway_gray_color_avg <= largest_area_color[2]):
        # roadway dang chiem nhieu dien tich nhat
        roadway_binary_image = imgproc.inRange1Channel(gray_image, largest_area_color[0], largest_area_color[2])
        roadway_binary_image = imgproc.erode(roadway_binary_image, (2, 2))
        roadway_binary_image = imgproc.findBiggestContour(roadway_binary_image)
        roadway_binary_image = imgproc.dilate(roadway_binary_image, (2, 2))

        shadow_binary_image = imgproc.findOtherArea(gray_image, 0, largest_area_color[0])
        shadow_binary_image = imgproc.erode(shadow_binary_image, (2, 2))
    elif (roadway_gray_color_avg < largest_area_color[0]):
        # dang huong ra ngoai
        shadow_binary_image = np.zeros_like(roadway_binary_image)
    else:
        # shadow dang chiem nhieu dien tich nhat
        shadow_binary_image = imgproc.inRange1Channel(gray_image, largest_area_color[0], largest_area_color[2])

        roadway_binary_image = imgproc.erode(roadway_binary_image, (2, 2))

    full_road = cv2.bitwise_or(roadway_binary_image, shadow_binary_image)
    full_road = imgproc.findBiggestContour(full_road)
    full_road = imgproc.erode(full_road, (5, 5))
    full_road = imgproc.findBiggestContour(full_road)
    full_road = imgproc.dilate(full_road, (5, 5))
    return full_road


def generator(batch_size):
    batch_features = np.zeros((batch_size, 1, 84,84))
    batch_segment = np.zeros((batch_size,1, 84, 84))
    while True:
        for i in range(batch_size):

            image_name = random.choice(image_paths)
            mat = cv2.imread(image_name)
            croped = mat[84:240, :]
            resized = cv2.resize(croped, (84, 84))
            gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

            batch_features[i] = [gray / 255]
            batch_segment[i] = [process_image(croped) / 255]
        yield batch_features, batch_segment

def generator_json(batch_size):
    batch_features = np.zeros((batch_size, 1, 84,84))
    batch_segment = np.zeros((batch_size,1, 84, 84))
    while True:
        for i in range(batch_size):

            image_name = random.choice(image_paths)
            json_path = image_name.replace('.png','.json')
            segment,gray = read_json(json_path)
            gray = cv2.resize(gray,(84,84))
            segment = cv2.resize(segment, (84, 84))
            segment = cv2.threshold(segment, 0.1, 1., cv2.THRESH_BINARY)[1]




            batch_features[i] = [gray]
            batch_segment[i] = [segment]
        yield batch_features, batch_segment


class My_Callback(Callback):

    def on_epoch_end(self, epoch, logs={}):
        image = cv2.imread(random.choice(image_paths),0)
        image = cv2.resize(image,(84,84))/255.

        y = self.model.predict(np.array([[image]]))
        image *= 255.
        y *= 255
        cv2.imwrite('/content/log/{}.jpg'.format(epoch), np.concatenate((image, y[0][0]), 1))
        self.model.save_weights('/content/model.h5')
        return


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


# while True:
#     x,y= next(generator(1))
#     image = x[0]
#     seg = y[0]
#     cv2.imshow('test',np.concatenate((image,seg),1))
#     cv2.waitKey(0)



model = Pun_net()
model.summary()
model.compile(loss='binary_crossentropy', optimizer='rmsprop',metrics=['binary_accuracy'])
#model.load_weights('/content/model.h5')
callback = My_Callback()
model.fit_generator(generator_json(32), steps_per_epoch=500, epochs=25, callbacks=[callback])
model.save_weights('/content/model.h5')


#TEST 

# cap = cv2.VideoCapture('/home/binhbumpro/Videos/outpy1.avi')
# out = None
# # detector = SSD_Detector('/home/binhbumpro/catkin_ws/src/rolling_thunder_python_node/scripts/object_graph')
# while True:
#     start_time = time.time()
#     ret, mat = cap.read()

#     # if not ret:
#     #     break
#     # # h,w,c = mat.shape
#     # if out is None:
#     #     out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 24.0, (84 * 2, 84))
#     mat =cv2.imread('test.png')
#     croped = mat[85:240, :]

#     resized = cv2.resize(croped, (84, 84))
#     gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY) / 255
#     # i_d = detector.detect_object(mat)
#     # i_d.visualize()
#     y = model.predict(np.array([[gray]]))
#     # gray *= 255.
#     # y *= 255
#     # print(image.shape,y[0][0].shape)
#     # print(y[0][0].mean())
#     demo = np.concatenate((gray, y[0][0]), 1)
#     print("FPS: ", 1.0 / (time.time() - start_time))
#     cv2.imshow('test', demo)
#     # cv2.imshow('original', i_d.image)
#     # out.write(demo)
#     cv2.waitKey(1)


