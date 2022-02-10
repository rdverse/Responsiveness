import re

import pandas as pd
import numpy as np
#import ast
import os
import glob
import tqdm
import matplotlib.pyplot as plt


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
FRAME_DIVIDER = 14
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


def _plot_presence(distances,personID, col_x):
    if not os.path.isdir('../keypointPlots'):
        os.mkdir('../keypointPlots')
    plt.plot(distances)
    plt.show()

def get_euclidean(personID, frames,keypointNo, col_x, col_y):
    dist = 0
    #print(frames)
    frameDenominator = 0
    distances = list()
    distances_r = list()


    for frame in range(1, MAX_FRAMES+1):
        
        if frame in frames:
            pos = frames.index(frame)
            if frames[pos] - frames[pos - 1] == 1:
                
                try:
                    curDist = np.sqrt((col_x[pos] - col_x[pos - 1])**2 +
                                          ((col_y[pos] - col_y[pos - 1])**2))
                    dist = dist + curDist

                    distances_r.append(dist)
                    distances.append(curDist)

                    frameDenominator += 1
                except:
                    pass
            else:
                distances.append(-2)
                distances_r.append( distances_r[-1] if len(distances_r)>0 else 0)
                
        else:

            distances.append(-1)
            distances_r.append(distances_r[-1] if len(distances_r)>0 else 0)
                
    try:

        dist = dist / frameDenominator
    except:
        dist = 0

    DISTANCES_TRACK[str(personID) + '_' + keypointNo] = distances
    DISTANCES_TRACK[str(personID) + '_' + keypointNo + '_' + 'r'] = distances_r

    return dist


if __name__=='__main__':
    path = 'chrisPP/'
    save_file_name = 'saved_values.csv'
    save_distance = 'saved_distances.csv'
    os.chdir(path)
    MAX_FRAMES = set_max_frames()

    if os.path.isfile(os.path.join(path, save_file_name)):
        os.remove(os.path.join(path, save_file_name))

    print('Generating file and calculating distance of each keypoint')
    save_value = pd.DataFrame(columns=['personID', 'keypoint', 'distance'])


    for person in glob.glob("*.csv"):
        personID = int(person.split('.')[0])
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

            pp = re.sub(r'\[[0-9\.\s]+\]', add_comma, pp)
            pp = re.sub(r'([0-9\.]+)', add_comma, pp)
            pp = eval(pp)

            #np.array([ast.literal_eval(p) for p in ast.literal_eval(pp)]).flatten()
            pp = np.array(pp)

            pp = np.take(pp, [0, 1], axis=1)
            pp = pp.flatten()

            calc_df.loc[len(calc_df)] = pp
            #Total frames detected
            #print(calc_df.head())

        for col in col_nos:
            distance = get_euclidean(personID, frames, str(col), calc_df[str(col) + '_x'],
                                    calc_df[str(col) + '_y'])

            save_value.loc[len(save_value)] = np.array(
                [str(personID), col, distance])


    #save values file
    save_value.to_csv(save_file_name, sep=',', encoding='utf-8')

    save_value = save_value.pivot(index='personID',
                                columns='keypoint',
                                values='distance')

    for col in save_value.columns:
        try:
            ch = "".join(COCO_KEYPOINT_INDEXES[col].split("_"))
        except:
            pass
        try:
            save_value.rename(columns={col: ch}, inplace=True)
        except:
            pass

    frameCount = list()
    #print(save_value.head())

    #Loop to add the frame counts as a new column
    for pID in save_value.index:
        frameCount.append(len(pd.read_csv(str(pID) + '.csv')))
    #Save distances file
    save_value['frameCount'] = frameCount

    COCO_KEYPOINT_INDEXES = {str(k):v for k,v in COCO_KEYPOINT_INDEXES.items()}
    save_value.rename(columns = { c: COCO_KEYPOINT_INDEXES[c] for c in save_value.columns if c in COCO_KEYPOINT_INDEXES.keys()}, inplace = True)

    save_value.to_csv(save_distance, sep=',', encoding='utf-8')
    
    DISTANCES_TRACK.to_csv("distances_track.csv")   


    #    frameCount = len(df)

    #    pID = person.split('.')[0]
    #   save_value['distance'] = [float(i) for i in save_value['distance']]
    #  save_value.distance.loc[
    #     save_value['personID'] ==
    #    pID] = save_value.distance[save_value['personID'] == pID] / frameCount

    #print(save_value)