#! /usr/bin/env python
"""
Created on 11/04/2016
@author: Sam Pfeiffer
Get depth of a pixel or pixel area.
"""
import algorithms

from algorithms.utils import *
from algorithms.rosws import *

import sys
import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import struct
import cv2
import numpy as np

# from http://answers.ros.org/question/90696/get-depth-from-kinect-sensor-in-gazebo-simulator/


def get_pixels_depth(ini_x, ini_y, end_x, end_y, depth_image=None):
    rospy.loginfo(
        "Getting average depth at: " + str((ini_x, ini_y, end_x, end_y)))
    # rospy.loginfo("Waiting for a depth image...")
    # img = rospy.wait_for_message('/xtion/depth/image_raw',
    #                              Image)
    img = depth_image
    rospy.loginfo("The image has size: " + str(img.width) +
                  " width, " + str(img.height) + " height")
    if ini_x < 0:
        rospy.logerr("Can't ask for a pixel depth out of image (ini_x < 0)")
        return None
    if ini_y < 0:
        rospy.logerr("Can't ask for a pixel depth out of image (ini_y < 0)")
        return None
    if end_x > img.width:
        rospy.logerr("Can't ask for a pixel depth out of image (end_x > img.width [%s])" % (
            str(img.width)))
        return None
    if end_y > img.height:
        rospy.logerr("Can't ask for a pixel depth out of image (end_x > img.height [%s])" % (
            str(img.height)))
        return None

    if (img.step / img.width) == 4:
        rospy.loginfo("Got a rectified depth image (4 byte floats)")
    else:
        rospy.loginfo("Got a raw depth image (2 byte integers)")

    # Compute the average of the area of interest
    sum_depth = 0
    pixel_count = 0
    nan_count = 0
    for x in range(ini_x, end_x):
        for y in range(ini_y, end_y):
            pixel = get_pixel_depth(x, y, img)
            # print "Curr pixel is: '" + str(pixel) + "' of type: " + str(type(pixel))
            if pixel != pixel:  # check if nan
                nan_count += 1
            else:
                sum_depth += pixel
                pixel_count += 1

    if pixel_count > 0:
        avg = sum_depth / float(pixel_count)
        rospy.loginfo("Average is: " + str(avg) + " (with " + str(pixel_count) +
                      " valid pixels, " + str(nan_count) + " NaNs)")

        return avg
    else:
        rospy.logwarn("No pixels that are not NaN, can't return an average")
        return None


def get_pixel_depth(x, y, depth_image=None):
    img = depth_image

    if x < 0:
        rospy.logerr("Can't ask for a pixel depth out of image (x < 0)")
        return None
    if y < 0:
        rospy.logerr("Can't ask for a pixel depth out of image (y < 0)")
        return None
    if x > img.width:
        rospy.logerr("Can't ask for a pixel depth out of image (x > img.width [%s])" % (
            str(img.width)))
        return None
    if y > img.height:
        rospy.logerr("Can't ask for a pixel depth out of image (x > img.height [%s])" % (
            str(img.height)))
        return None

    index = (y * img.step) + (x * (img.step / img.width))

    if (img.step / img.width) == 4:
        # rospy.loginfo("Got a rectified depth image (4 byte floats)")
        byte_data = ""
        for i in range(0, 4):
            byte_data += img.data[index + i]

        distance = struct.unpack('f', byte_data)[0]
        return distance
    else:
        # rospy.loginfo("Got a raw depth image (2 byte integers) (UNTESTED)")
        # Expecting the data to be an unsigned short representing mm
        if img.is_bigendian:
            distance = struct.unpack('>H', img.data[index:index + 2])[0]
        else:

            distance = struct.unpack('<H', img.data[index:index + 2])[0]
        return distance


def from_compressed_image_to_image(compressed_image, bridge=None):
    # compressed_image must be from a topic compressedDepth (not just compressed)
    # as it's encoded in PNG
    # Code from: https://answers.ros.org/question/249775/display-compresseddepth-image-python-cv2/
    msg = compressed_image
    depth_fmt, compr_type = msg.format.split(';')
    # remove white space
    depth_fmt = depth_fmt.strip()
    compr_type = compr_type.strip()
    if compr_type != "compressedDepth":
        raise Exception("Compression type is not 'compressedDepth'."
                        "You probably subscribed to the wrong topic.")

    # remove header from raw data
    depth_header_size = 12
    raw_data = msg.data[depth_header_size:]

    depth_img_raw = cv2.imdecode(np.frombuffer(raw_data, np.uint8),
                                 # the cv2.CV_LOAD_IMAGE_UNCHANGED has been removed
                                 -1)  # cv2.CV_LOAD_IMAGE_UNCHANGED)
    if depth_img_raw is None:
        # probably wrong header size
        raise Exception("Could not decode compressed depth image."
                        "You may need to change 'depth_header_size'!")

    if not bridge:
        bridge = CvBridge()
    return bridge.cv2_to_imgmsg(depth_img_raw, "mono16")

def processing_depth(depth_raw):
    
    np_app = np.fromstring(depth_raw.data,np.uint8)
            
    frame_depth = cv2.imdecode(np_app,cv2.IMREAD_COLOR)  
            
    bridge = CvBridge()
            
    depth_raw = bridge.cv2_to_imgmsg(frame_depth,"8UC3")

    return frame_depth,depth_raw
