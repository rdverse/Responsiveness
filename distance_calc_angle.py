import re
from importlib_metadata import DistributionFinder

import pandas as pd
import numpy as np
#import ast
import os
import glob
import tqdm
import matplotlib.pyplot as plt

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

COCO_KEYPOINT_NAMES =  {v: k for k, v in COCO_KEYPOINT_INDEXES.items()}
CKN = COCO_KEYPOINT_NAMES

MAX_FRAMES = 0 
FRAME_DIVIDER = 1
DISTANCES_TRACK = pd.DataFrame()

_selectColsLeft = []
_selectColsRight = []

colsLeft = [[CKN["left_shoulder"], CKN["left_elbow"], CKN["left_wrist"]],
            [CKN["nose"],CKN["left_shoulder"], CKN["left_wrist"]],
            [CKN["nose"],CKN["left_shoulder"], CKN["left_hip"]],
            [CKN["left_shoulder"],CKN["left_hip"], CKN["left_knee"]]
            ]

colsRight = [[CKN["right_shoulder"], CKN["right_elbow"], CKN["right_wrist"]],
            [CKN["nose"],CKN["right_shoulder"], CKN["right_wrist"]],
            [CKN["nose"],CKN["right_shoulder"], CKN["right_hip"]],
            [CKN["right_shoulder"],CKN["right_hip"], CKN["right_knee"]]
            ]

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

def distance_between(handData,pos):
    #Initialize dist ls and dist rs lists
    dists = {
        "left":[],
        "right":[]
    }
    combinations={"left":[_selectColsLeft[:2],_selectColsLeft[1:]], 
                "right":[_selectColsRight[:2],_selectColsRight[1:]]}
    for key,val in combinations.items():
        for limbIndex in val:
            xa,xb = handData[key]["x"][[str(no) + '_x' for no in limbIndex]].iloc[pos].values
            ya,yb = handData[key]["y"][[str(no) + '_y' for no in limbIndex]].iloc[pos].values
            distBetween = calc_dist(xa,ya,xb,yb)
            dists[key].append(distBetween)
    # print(dists)
    return dists["left"], dists["right"]


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
    curDistL, curDistR, prevDistL, prevDistR, = [0,0],[0,0],[0,0],[0,0]
    
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
                curDistL,curDistR = distance_between(handData,pos-1)
                
                deltaDistLs.append(abs(prevDistL[0] - curDistL[0])
                                    + abs(prevDistL[1] - curDistL[1]))
                deltaDistRs.append(abs(prevDistR[0] - curDistR[0]) + 
                                    abs(prevDistR[1] - curDistR[1]))
                
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

    # save_value = pd.DataFrame(columns=['personID', 'keypoint', 'distance'])

    personAngles = {"personID":[]}

    
    for person in glob.glob("*.csv"):
        saveDF= pd.DataFrame()
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

        personAngles["personID"].append(personID)
        for _, row in tqdm.tqdm(df.iterrows()):        
            pp = row['pose_preds']
            if pp==0 or pp=='0':
                continue
            pp = eval(pp)
            pp = np.array(pp)
            pp = np.take(pp, [0,1], axis=1)
            pp = pp.flatten()
            calc_df.loc[len(calc_df)] = pp

        # Filter data for each iteration
        for i in range(len(colsLeft)):
            _selectColsLeft = colsLeft[i]
            _selectColsRight = colsRight[i]

            handData = {"left" : {
                "x" : calc_df[ [str(col) + '_x' for col in _selectColsLeft]],
                "y" : calc_df[ [str(col) + '_y' for col in _selectColsLeft]]
            },

            "right": {
                "x":calc_df[[str(col) + '_x' for col in _selectColsRight]],
                "y":calc_df[[str(col) + '_y' for col in _selectColsRight]]}
            }

            # angleLs and angleRs are lists now where they store
            # a list of angles defined for each side
            angleL,angleR, distL, distR, nframe = get_measures(frames, handData)

            if "thetaL" + str(i) not in personAngles.keys():
                personAngles["thetaL" + str(i)] = list()   
                personAngles["thetaR"+ str(i)] =  list()
                personAngles["distL"+ str(i)] = list()
                personAngles["distR"+ str(i)] = list()
            
            # Sum of theta and distance per person
            personAngles["thetaL" + str(i)].append(np.sum(angleL)/nframe)
            personAngles["thetaR"+ str(i)].append(np.sum(angleR)/nframe)
            personAngles["distL"+ str(i)].append(np.sum(distL)/nframe)
            personAngles["distR"+ str(i)].append(np.sum(distR)/nframe)
            
            # history of angles and distances per person
            saveDF["thetaL" + str(i)] = angleL
            saveDF["thetaR"+ str(i)] = angleR
            saveDF["distL"+ str(i)] = distL
            saveDF["distR"+ str(i)] = distR

        personAngles["theta"] = list(np.sum([value for key,value in personAngles.items() if key[:-2]=="theta"],axis=0))
        personAngles["dist"] = list(np.sum([value for key,value in personAngles.items() if key[:-2]=="dist"],axis=0)) 

        if not os.path.isdir('angles'):
            os.mkdir('angles')

        saveDF["nose_x"] = calc_df["0_x"].values[:len(angleL)]
        saveDF["nose_y"] = calc_df["0_y"].values[:len(angleL)]
        saveDF.to_csv("angles/" + str(personID) + 'a' + '.csv') 
               #For indivisual person
        # saveDF = pd.DataFrame(np.hstack((angleLs.reshape(-1,1),
        #                         angleRs.reshape(-1,1),
        #                         distLs.reshape(-1,1),
        #                         distRs.reshape(-1,1)
        #                         )),
        #                         columns=["thetaL", 
        #                                 "thetaR", 
        #                                 "distL",
        #                                 "distR",
        #                                  "nose_x", 
        #                                  "nose_y"])
        
        # angleRs.reshape(-1,1) + angleLs.reshape(-1,1),
        #                         distLs.reshape(-1,1) + distRs.reshape(-1,1),
        #                         nose_x.reshape(-1,1),
        #                         nose_y.reshape(-1,1)

        # Individual persons dataframe

    #This is the final dataframe to be saved
    pd.DataFrame.from_dict(personAngles).to_csv("angles/personAngles.csv")