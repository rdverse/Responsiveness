import re

import pandas as pd
import numpy as np
#import ast
import os
import glob
import tqdm
import matplotlib.pyplot as plt

import cv2
import cv2
# from skimage.measure import compare_ssim
from math import atan2, degrees


COCO_KEYPOINT_INDEXES = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}


MAX_FRAMES = 0 
FRAME_DIVIDER = 1
DISTANCES_TRACK = pd.DataFrame()

def set_max_frames():
    folders = os.listdir() 
    allFileIndices = list()
    for root, dirs, files in os.walk('.'):
        print(root)
        try:
            files = [int(f.strip('.jpg')) for f in files]
            allFileIndices.extend(files)
        except:
            pass
        
    return(int(max(allFileIndices)/FRAME_DIVIDER))

def add_comma(match):
    return match.group(0) + ','


#Divide by maximum value
def normalize(save_value):
    for col in save_value.columns:
        #for val in save_value[col]:
        print('{} , {}'.format(col, np.max(save_value[col])))
        save_value[col] = save_value[col].astype(float)
        save_value[col] = save_value[col] / np.max(save_value[col])
    return save_value

# def _plot_presence(distances,personID, col_x):
#     if not os.path.isdir('../keypointPlots'):
#         os.mkdir('../keypointPlots')
#     plt.plot(distances)
#     plt.show()

def calc_dist(xa, ya, xb, yb):
    dist = np.sqrt((xa - xb)**2 + ((ya - yb)**2))
    return dist 

def distance_between(frames, handData):
    comb = [[],[]]
    for comb in combinations:
        [[xa,ya],[xb,yb]] = comb
        distBetween = calc_dist(xa,ya,xb,yb)
        distBetween+=distBetween
    # return deg2 - deg1 if deg1 <= deg2 else 360 - (deg1 - deg2)
    return distLs, distRs, nframes


# def distance_between(handDistance, pos):
def angle_between(handAngle,pos):

    x1, x2, x3 = handAngle["x"].iloc[pos].values.T*1920
    y1, y2, y3 = handAngle["y"].iloc[pos].values.T*1080

    deg1 = (360 + degrees(atan2(x1 - x2, y1 - y2))) % 360
    deg2 = (360 + degrees(atan2(x3 - x2, y3 - y2))) % 360
    return deg2 - deg1 if deg1 <= deg2 else 360 - (deg1 - deg2)


def get_measures(frames, handData):
    dist = 0
    #print(frames)
    nframes = 0
    # resultant angles 
    deltaTheta = list()
    # colHand_x, colHand_y
    curThetaL, curThetaR, prevThetaL, prevThetaR, = np.zeros(4)
    curDistL, curDistR, prevDistL, prevDistR, = np.zeros(4)
    
    deltaThetaLs,deltaThetaRs = list(),list()
    deltaDistLs,deltaDistRs = list(),list()

    print(MAX_FRAMES)
    for frame in range(1, MAX_FRAMES+1):
        #print(frames)
        if frame in frames:
            pos = frames.index(frame) 
            if frames[pos] - frames[pos - 1] == 1:
                
                ## Angles
                curThetaL = angle_between(handData["left"],pos-1)
                curThetaR = angle_between(handData["right"],pos-1)

                deltaThetaLs.append(abs(prevThetaL - curThetaL))
                deltaThetaRs.append(abs(prevThetaR - curThetaR))
                
                prevThetaL = curThetaL
                prevThetaR = curThetaR
                
                # Distances
                curDistL = distance_between(handData["left"],pos-1)
                curDistR = distance_between(handData["right"],pos-1)

                deltaDistLs.append(abs(prevDistL - curDistL))
                deltaDistRs.append(abs(prevDistR - curDistR))
                
                prevDistL = curDistL
                prevDistR = curDistR
                
                nframes += 1
    
    return np.array(deltaThetaLs), np.array(deltaThetaRs),\
        np.array(deltaDistLs), np.array(deltaDistRs), nframes

if __name__=='__main__':

    try:
        os.mkdir("angles")
    except:
        print("Angles folder is already present")
        pass

    path = 'chrisPP/'
    save_file_name = 'saved_values.csv'
    save_distance = 'saved_distances.csv'
    os.chdir(path)
    MAX_FRAMES = set_max_frames()

    if os.path.isfile(os.path.join(path, save_file_name)):
        os.remove(os.path.join(path, save_file_name))

    print('Generating file and calculating distance of each keypoint')
    save_value = pd.DataFrame(columns=['personID', 'keypoint', 'distance'])

    personAngles = {"personID":[], "thetaL":[], "thetaR":[]}
    for person in glob.glob("*.csv"):
        personID = int(person.strip(".csv"))

        print(personID)
        print('Person : %d' % int(personID))
        df = pd.read_csv(person)  #[2:]

        col_nos = list(range(17))
        col_mod = [[str(col) + '_x', str(col) + '_y'] for col in col_nos]
        col_mod = np.array(col_mod).flatten()

        calc_df = pd.DataFrame(columns=col_mod)
        #Store the frame numbers in this
        frames = list(df.frame)
        frames = [frame / FRAME_DIVIDER for frame in frames]
        for _, row in tqdm.tqdm(df.iterrows()):        
            # print((row[1]))
            pp = row['pose_preds']
            # print("pp is here")
            if pp==0 or pp=='0':
                continue
            pp = eval(pp)
            pp = np.array(pp)
            # print(pp.shape)
            pp = np.take(pp, [0,1], axis=1)
            pp = pp.flatten()
            calc_df.loc[len(calc_df)] = pp

        handColsLeft = [5,7,9]
        handColsRight = [6,8,10]

        handData = {"left" : {
            "x" : calc_df[ [str(col) + '_x' for col in handColsLeft]],
            "y" : calc_df[ [str(col) + '_y' for col in handColsLeft]]
        },

        "right": {
            "x":calc_df[[str(col) + '_x' for col in handColsRight]],
            "y":calc_df[[str(col) + '_y' for col in handColsRight]]}
        }

        # angleLs,angleRs, nframes = get_angle(frames, handData)
        # distLs, distRs, nframes = get_euclidean(frames, handData)
        print(frames)
        angleLs,angleRs, distLs, distRs, nframes = get_measures(frames, handData)


        # print(angleL,angleR)
        personAngles["personID"].append(personID)
        personAngles["thetaL"].append(np.sum(angleLs)/nframes)
        personAngles["thetaR"].append(np.sum(angleRs)/nframes)
        personAngles["theta"] = list(np.array(personAngles["thetaR"]) + np.array(personAngles["thetaL"])) 


        if not os.path.isdir('angles'):
            os.mkdir('angles')

        pd.DataFrame(np.hstack((angleLs.reshape(-1,1),
                                angleRs.reshape(-1,1),
                                angleRs.reshape(-1,1) + angleLs.reshape(-1,1),
                                )),
                                columns=["thetaL", "thetaR", "theta"]).to_csv(
                                    "angles/" + str(personID) + 'a' + '.csv')

# .detach().cpu().numpy()
    pd.DataFrame.from_dict(personAngles).to_csv("angles/personAngles.csv")
