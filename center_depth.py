import os
import pathlib
import cv2
import time
import argparse
import math
import torch

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


def load_midas():

    model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    #model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    #model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform


    return midas, transform,device



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

# load tensorflow hub model
# model_handle = 'CenterNet HourGlass104 Keypoints 512x512'
model_handle = 'CenterNet HourGlass104 Keypoints 1024x1024'

print('loading model...')
hub_model = hub.load(ALL_MODELS[model_handle])
print('model loaded!')
'''
Function for adding arguments through argparse
'''

depthModel, depthTransform, device = load_midas()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v[0].lower() == 't':
        return True
    elif v[1].lower() == 'f':
        return False


def get_argparser():
    parser = argparse.ArgumentParser(
        description='CenterNet for keypoints detection', add_help=True)
    parser.add_argument(
        "--initialize",
        default=False,
        type=str2bool,
        help='Populate all files to group directory ? (True or False)')

    parser.add_argument(
        "--video_input",
        help='Just add the name of the video file in box_data dir')

    parser.add_argument("--video_output",
                        default=True,
                        type=str2bool,
                        help="Do you want video output ? (True or False)")
    return parser


def get_detections(image_np):
    # running inference
    results = hub_model(image_np)

    # different object detection models have additional results
    # all of them are explained in the documentation
    result = {key: value.numpy() for key, value in results.items()}

    label_id_offset = 0
    image_np_with_detections = image_np.copy()

    # Use keypoints if available in detections
    keypoints, keypoint_scores = None, None
    if 'detection_keypoints' in result:
        keypoints = result['detection_keypoints'][0]
        keypoint_scores = result['detection_keypoint_scores'][0]

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections[0],
        result['detection_boxes'][0],
        (result['detection_classes'][0] + label_id_offset).astype(int),
        result['detection_scores'][0],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=20,
        min_score_thresh=.30,
        agnostic_mode=False,
        keypoints=keypoints,
        keypoint_scores=keypoint_scores,
        keypoint_edges=COCO17_HUMAN_POSE_KEYPOINTS)

    return (image_np_with_detections, result)


args = get_argparser().parse_args()

if args.initialize:
    multiplier = 1
else:
    multiplier = 1

groupPATH = 'groupPATH'
#Directory setup
if not os.path.isdir(groupPATH):
    os.mkdir(groupPATH)

# Other static declarations
cnt = 0
vidFile = os.path.join('box_data', args.video_input)

cap = cv2.VideoCapture(vidFile)
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (frameWidth, frameHeight)
codec = cv2.VideoWriter_fourcc(*'DIVX')
fps = int(cap.get(cv2.CAP_PROP_FPS))

frame_rate_divider = int(fps * multiplier)

# change these two to alter the fps of read and write
frame_rate_divider=1 # 1  fps
multiplier = 1/fps #1/fps 1

output = cv2.VideoWriter('huma.avi', codec, int(1 / multiplier), size)
outputDepth = cv2.VideoWriter('humaDepth.avi', codec, int(1 / multiplier), size)

# read the video

def norm_depth(d):
    d = d-35
    d = abs(d)
    d = d/35
    return(d)
inp = []
while (cap.isOpened()):
    stime = time.time()

    #try:
    ret, frame = cap.read()
    image_np = np.array([frame.copy()])
        # print("image_Np shape")
        # print(image_np.shape)
    #except:
    #    continue

    if ret:
        if cnt % frame_rate_divider == 0:
            frame_keypoints, results = get_detections(image_np)
            frame_depth = depthTransform(image_np).to(device)
            frame_depth= depthModel(frame_depth)
            # for one image
            imgList = list()
            #Go through the masked images and save them in a folder
            results = {key: val[0] for key, val in results.items()}
            results["scores"] = results.pop("detection_scores")
            results["keypoints"] = results.pop("detection_keypoints")
            results["keypoints"] = [[Px,Py,frame_depth[Py][Px]] for [Px,Py] in results["keypoints"]]

            results_copy = results.copy()
            keepIndices = [i for i,score in enumerate(results['scores']) if score>0.3]
            
            #filter results that have a low threshold/confidence in prediction
            for key, val in results.items():
                if key!='num_detections':
                    results[key]=list(np.array(results[key])[keepIndices])

            keepIndices = [i for i,val in enumerate(results['detection_classes']) if val==1.0]
            
            for key, val in results.items():
                if key!='num_detections':
                    results[key]=np.array(results[key])[keepIndices]

            results["num_detections"] = len(results["keypoints"])
            for no in range(int(results["num_detections"])):
                box = results["detection_boxes"][no]
                image_name = os.path.join(
                    groupPATH,
                    str(no) + '_' + str(cnt) + '.jpg')

                # normalize the keypoints as well
                results['keypoints'][no] = [[pk[0]*frameWidth, pk[1]*frameHeight] for pk in results['keypoints'][no]]
                
                # print("BOX")
                # print(box)
                # multiply the box coordinates with frame width and height
                Y1, X1, Y2, X2 = tuple(box)
            results['images'] = imgList

            if not args.initialize:
                if cnt == 0:
                    #This sets up the csv and directories in the mainPath
                    detector = DetectionSetupMode()
                else:
                    detector.classifyDetections(results, cnt)

            if args.video_output:
                output.write(np.squeeze(frame_keypoints))
                output.write(np.squeeze(frame_depth))
            print(cnt)
            print('Time elapsed in the video : {} minutes'.format(
                (cnt / (fps * 60))))
            cnt += 1
        else:
            cnt += 1
    else:
        break
        print("stuck here?")

cap.release()
output.release()