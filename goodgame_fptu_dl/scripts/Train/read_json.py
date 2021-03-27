import argparse
import base64
import glob
import json
import os
import random
import sys

import cv2
import matplotlib.pyplot as plt

from labelme import utils

PY2 = sys.version_info[0] == 2


def read_json(json_file):
    data = json.load(open(json_file))

    if data['imageData']:
        imageData = data['imageData']
    else:
        imagePath = os.path.join(os.path.dirname(json_file), data['imagePath'])
        with open(imagePath, 'rb') as f:
            imageData = f.read()
            imageData = base64.b64encode(imageData).decode('utf-8')
    img = utils.img_b64_to_arr(imageData)

    label_name_to_value = {'_background_': 0}
    for shape in sorted(data['shapes'], key=lambda x: x['label']):
        label_name = shape['label']
        if label_name in label_name_to_value:
            label_value = label_name_to_value[label_name]
        else:
            label_value = len(label_name_to_value)
            label_name_to_value[label_name] = label_value
    lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)
    return lbl*1.,cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)/255.


if __name__ == '__main__':
    json_paths = glob.glob('/home/ubuntu/datacds/segmentation/**/*.json')
    while True:
        mat,img = read_json(random.choice(json_paths))
        # print(img.mean())
        mat = cv2.resize(mat,(84,84))
        mat = cv2.threshold(mat, 0.1, 1., cv2.THRESH_BINARY)[1]
        print(set(list(mat.flatten())))
        cv2.imshow('test', mat)
        cv2.imshow('tes2t', img)
        k  = cv2.waitKey(0)
        if k == ord('q'):
            break