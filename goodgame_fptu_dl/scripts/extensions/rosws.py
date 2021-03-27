#!/usr/bin/env python
# Note that this needs:
# sudo pip install websocket-client
# not the library called 'websocket'

import json
from uuid import uuid4
import websocket
import yaml
from geometry_msgs.msg import  PoseStamped
import rospy
from std_msgs.msg import Header,String,Float32,Int8
from sensor_msgs.msg import CompressedImage,Image
from rospy_message_converter import message_converter
import cv2
import numpy as np

class WebsocketROSClient(object):
    def __init__(self, websocket_ip, port=9090):
        """
        Class to manage publishing to ROS thru a rosbridge websocket.
        :param str websocket_ip: IP of the machine with the rosbridge server.
        :param int port: Port of the websocket server, defaults to 9090.
        """
        #print("Connecting to websocket: {}:{}".format(websocket_ip, port))
        
        self.ws = websocket.create_connection(
            'ws://' + websocket_ip + ':' + str(port))
        self._advertise_dict = {}

    def _advertise(self, topic_name, topic_type):
        """
        Advertise a topic with it's type in 'package/Message' format.
        :param str topic_name: ROS topic name.
        :param str topic_type: ROS topic type, e.g. std_msgs/String.
        :returns str: ID to de-advertise later on.
        """
        new_uuid = str(uuid4())
        self._advertise_dict[new_uuid] = {'topic_name': topic_name,
                                          'topic_type': topic_type}
        advertise_msg = {"op": "advertise",
                         "id": new_uuid,
                         "topic": topic_name,
                         "type": topic_type
                         }
        self.ws.send(json.dumps(advertise_msg))
        return new_uuid

    def _unadvertise(self, uuid):
        unad_msg = {"op": "unadvertise",
                    "id": uuid,
                    # "topic": topic_name
                    }
        self.ws.send(json.dumps(unad_msg))
        
    def __del__(self):
        """Cleanup all advertisings"""
        d = self._advertise_dict
        for k in d:
            self._unadvertise(k)

    def _publish(self, topic_name, message):
        """
        Publish onto the already advertised topic the msg in the shape of
        a Python dict.
        :param str topic_name: ROS topic name.
        :param dict msg: Dictionary containing the definition of the message.
        """
        msg = {
            'op': 'publish',
            'topic': topic_name,
            'msg': message
        }
        json_msg = json.dumps(msg)
        self.ws.send(json_msg)

    def publish(self, topic_name, ros_message):
        """
        Publish on a topic given ROS message thru rosbridge.
        :param str topic_name: ROS topic name.
        :param * ros_message: Any ROS message instance, e.g. LaserScan()
            from sensor_msgs/LaserScan.
        """
        # First check if we already advertised the topic
        d = self._advertise_dict
        for k in d:
            if d[k]['topic_name'] == topic_name:
                # Already advertised, do nothing
                break
        else:
            # Not advertised, so we advertise
            topic_type = ros_message._type
            self._advertise(topic_name, topic_type)
        # Converting ROS message to a dictionary thru YAML
        ros_message_as_dict = yaml.load(ros_message.__str__(), Loader=yaml.FullLoader)
        # Publishing
        self._publish(topic_name, ros_message_as_dict)

    def subscribe(self,topic_name, ros_message):
        # First check if we already advertised the topic
        d = self._advertise_dict
        for k in d:
            if d[k]['topic_name'] == topic_name:
                # Already advertised, do nothing
                break
        else:
            # Not advertised, so we advertise
            topic_type = ros_message._type
            self._advertise(topic_name, topic_type)
        # Converting ROS message to a dictionary thru YAML
        ros_message_as_dict = yaml.load(ros_message.__str__(), Loader=yaml.FullLoader)
        # Publishing
        return self._subscribe(topic_name, ros_message_as_dict, ros_message._type)

    def _subscribe(self, topic_name, message, type):
        """
        Publish onto the already advertised topic the msg in the shape of
        a Python dict.
        :param str topic_name: ROS topic name.
        :param dict msg: Dictionary containing the definition of the message.
        """
        msg = {
            'op': 'subscribe',
            'topic': topic_name,
            'type' : type
        }
        json_msg = json.dumps(msg)
        self.ws.send(json_msg)
        json_message = self.ws.recv()

        dictionary = json.loads(json_message)['msg']
        result =  message_converter.convert_dictionary_to_ros_message(type, dictionary)
        #print("Type: '%s' \n Received: '%s'" % (type, result))
        return result

#if __name__ == '__main__':
#    connect = WebsocketROSClient('127.0.0.1')
    
