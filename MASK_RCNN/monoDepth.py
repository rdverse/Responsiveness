import os
import pathlib
import cv2
import time
import argparse
import math

import matplotlib
import matplotlib.pyplot as plt

import io
import scipy.misc
import numpy as np
import pandas as pd
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from six.moves.urllib.request import urlopen

import tensorflow as tf
import tensorflow_hub as hub

tf.get_logger().setLevel('ERROR')

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import ops as utils_ops

from center_utils import *
from detection import DetectionSetupMode, compareData, Person
results_copy = []
PATH_TO_LABELS = './models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(
    PATH_TO_LABELS, use_display_name=True)

ROOT_DIR = os.getcwd()
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "mylogs")
# Local path to trained weights file

COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco_humanpose.h5")

#import tensorflow as tf
#physical_devices = tf.config.enxperimental.list_phyiscal_device('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)
os.environ['TFHUB_CACHE_DIR'] = 'cache'

# # load tensorflow hub model
# # model_handle = 'CenterNet HourGlass104 Keypoints 512x512'
# model_handle = 'CenterNet HourGlass104 Keypoints 1024x1024'

# print('loading model...')
# hub_model = hub.load(ALL_MODELS[model_handle])
# print('model loaded!')



import cv2
import numpy as np
import urllib.request
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as hub

url, filename = ("https://github.com/intel-isl/MiDaS/releases/download/v2/dog.jpg", "dog.jpg")
urllib.request.urlretrieve(url, filename)

# the runtime initialization will not allocate all memory on the device to avoid out of GPU memory
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     #tf.config.experimental.set_memory_growth(gpu, True)
#     tf.config.experimental.set_virtual_device_configuration(gpu,
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])

# input
img = cv2.imread('dog.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

img_resized = tf.image.resize(img, [384,384], method='bicubic', preserve_aspect_ratio=False)
img_resized = tf.transpose(img_resized, [2, 0, 1])
img_input = img_resized.numpy()
reshape_img = img_input.reshape(1,3,384,384)
tensor = tf.convert_to_tensor(reshape_img, dtype=tf.float32)

# load model
module = hub.load("https://tfhub.dev/intel/midas/v2/2", tags=['serve'])
output = module.signatures['serving_default'](tensor)
prediction = output['default'].numpy()
prediction = prediction.reshape(384, 384)
             
# output file
prediction = cv2.resize(prediction, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
print(" Write image to: output.png")
depth_min = prediction.min()
depth_max = prediction.max()
img_out = (255 * (prediction - depth_min) / (depth_max - depth_min)).astype("uint8")

cv2.imwrite("output.png", img_out)
plt.imshow(img_out)