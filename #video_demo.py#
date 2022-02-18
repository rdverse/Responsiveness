import os
import math
import numpy as np
import coco
import model as modellib
import visualize
from model import log
import cv2
import time
import argparse
import tqdm
import pdb

from detection import DetectionSetupMode, compareData, Person

ROOT_DIR = os.getcwd()
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "mylogs")
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco_humanpose.h5")

#import tensorflow as tf
#physical_devices = tf.config.enxperimental.list_phyiscal_device('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)


class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    KEYPOINT_MASK_POOL_SIZE = 7


inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights

model_path = os.path.join(ROOT_DIR, "mask_rcnn_coco_humanpose.h5")
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)
class_names = ['BG', 'person']


def cv2_display_keypoint(image,
                         boxes,
                         keypoints,
                         masks,
                         class_ids,
                         scores,
                         class_names,
                         skeleton=inference_config.LIMBS):
    # Number of persons
    N = boxes.shape[0]
    if not N:
        print("\n*** No persons to display *** \n")
    else:
        assert N == keypoints.shape[0] and N == class_ids.shape[0] and N==scores.shape[0],\
            "shape must match: boxes,keypoints,class_ids, scores"
    colors = visualize.random_colors(N)
    for i in range(N):
        color = colors[i]
        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
        for Joint in keypoints[i]:
            if (Joint[2] != 0):
                cv2.circle(image, (Joint[0], Joint[1]), 2, color, -1)

        #draw skeleton connection
        limb_colors = [[0, 0, 255], [0, 170, 255], [0, 255, 170], [0, 255, 0],
                       [170, 255, 0], [255, 170, 0], [255, 0, 0],
                       [255, 0, 170], [170, 0, 255], [170, 170, 0],
                       [170, 0, 170]]
        if (len(skeleton)):
            skeleton = np.reshape(skeleton, (-1, 2))
            neck = np.array(
                (keypoints[i, 5, :] + keypoints[i, 6, :]) / 2).astype(int)
            if (keypoints[i, 5, 2] == 0 or keypoints[i, 6, 2] == 0):
                neck = [0, 0, 0]
            limb_index = -1
            for limb in skeleton:
                limb_index += 1
                start_index, end_index = limb  # connection joint index from 0 to 16
                if (start_index == -1):
                    Joint_start = neck
                else:
                    Joint_start = keypoints[i][start_index]
                if (end_index == -1):
                    Joint_end = neck
                else:
                    Joint_end = keypoints[i][end_index]
                # both are Annotated
                # Joint:(x,y,v)
                if ((Joint_start[2] != 0) & (Joint_end[2] != 0)):
                    cv2.line(image, tuple(Joint_start[:2]),
                             tuple(Joint_end[:2]), limb_colors[limb_index], 3)
        mask = masks[:, :, i]
        image = visualize.apply_mask(image, mask, color)
        caption = "{} {:.3f}".format(class_names[class_ids[i]], scores[i])
        cv2.putText(image, caption, (x1 + 5, y1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
    return image


'''
Function for adding arguments through argparse

'''


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v[0].lower() == 't':
        return True
    elif v[1].lower() == 'f':
        return False


def get_argparser():
    parser = argparse.ArgumentParser(
        description='MaskRCNN for keypoints and mask detection', add_help=True)
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


args = get_argparser().parse_args()

if args.initialize:
    multiplier = 5
else:
    multiplier = 0.5

vidFile = os.path.join('box_data', args.video_input)

cap = cv2.VideoCapture(vidFile)

size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
codec = cv2.VideoWriter_fourcc(*'DIVX')
fps = int(cap.get(cv2.CAP_PROP_FPS))
groupPATH = 'groupPATH'

output = cv2.VideoWriter('human2.avi', codec, int(1 / multiplier), size)

if not os.path.isdir(groupPATH):
    os.mkdir(groupPATH)

cnt = 0
frame_rate_divider = int(fps * multiplier)

while (cap.isOpened()):
    stime = time.time()

    try:
        ret, frame = cap.read()
    except:
        continue

    if ret:
        if cnt % frame_rate_divider == 0:
            results = model.detect_keypoint([frame], verbose=0)
            r = results[0]
            # for one image
        # log("rois", r['rois'])
        # log("keypoints", r['keypoints'])
        #log("class_ids", r['class_ids'])
        #log("keypoints", r['keypoints'])
        #log("masks", r['masks'])
        #log("scores", r['scores'])

            imgList = list()
            #Go through the masked images and save them in a folder
            for no, maskedimage in enumerate(r['masks'].T):
                image_name = os.path.join(groupPATH,
                                          str(no) + '_' + str(cnt) + '.jpg')
                frame_copy = frame.copy()
                maskedimage = maskedimage.T

                for c in range(3):

                    frame_copy[:, :, c] = np.where(maskedimage == 1,
                                                   frame[:, :, c], 0)

                Y1, X1, Y2, X2 = tuple(r['rois'][no])

                frame_copy = frame_copy[math.floor(Y1):math.ceil(Y2),
                                        math.floor(X1):math.ceil(X2)]

                imgList.append(frame_copy)
                if args.initialize:
                    cv2.imwrite(image_name, frame_copy)

            r['images'] = imgList
            result_frame = cv2_display_keypoint(frame, r['rois'],
                                                r['keypoints'], r['masks'],
                                                r['class_ids'], r['scores'],
                                                class_names)

            # pdb.set_trace()

            if not args.initialize:
                if cnt == 0:
                    #This sets up the csv and directories in the mainPath
                    detector = DetectionSetupMode()
                else:
                    detector.classifyDetections(r, cnt)

            if args.video_output:
                output.write(result_frame)

            print(cnt)
            print('Time elapsed in the video : {} minutes'.format(
                (cnt / (fps * 60))))
            cnt += 1

        else:
            cnt += 1
    # print('FPS {:.1f}'.format(1 / (time.time() - stime)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
output.release()
cv2.destroyAllWindows()
