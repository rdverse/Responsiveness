import re

import pandas as pd
import numpy as np
#import ast
import os
import glob
import tqdm

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


def get_euclidean(col_name, col_x, col_y, frames):
    dist = 0
    #print(frames)
    frameDenominator = 0

    for e, i in enumerate(range(len(col_x))):
        if e > 0:
            if frames[e] - frames[e - 1] == 1:
                try:
                    dist = dist + np.sqrt((col_x[i] - col_x[i - 1])**2 +
                                          ((col_y[i] - col_y[i - 1])**2))
                    frameDenominator += 1
                except:
                    pass
    try:

        dist = dist / frameDenominator
    except:
        dist = 0
    return dist


path = 'chrisPP/'
save_file_name = 'saved_values.csv'
save_distance = 'saved_distances.csv'
os.chdir(path)

if os.path.isfile(os.path.join(path, save_file_name)):
    os.remove(save_file_name)

print('Generating file and calculating distance of each keypoint')
save_value = pd.DataFrame(columns=['personID', 'keypoint', 'distance'])


def add_comma(match):
    return match.group(0) + ','


for person in glob.glob("*.csv"):
    print('Person : %d' % int(person[0]))
    df = pd.read_csv(person)  #[2:]

    col_nos = list(range(17))
    col_mod = [[str(col) + '_x', str(col) + '_y'] for col in col_nos]
    col_mod = np.array(col_mod).flatten()

    calc_df = pd.DataFrame(columns=col_mod)
    #Store the frame numbers in this

    frames = list(df.frame)

    frames = [frame / 28 for frame in frames]

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
        distance = get_euclidean(0, frames, calc_df[str(col) + '_x'],
                                 calc_df[str(col) + '_y'])

        save_value.loc[len(save_value)] = np.array([person[0], col, distance])
#    frameCount = len(df)

#    pID = person.split('.')[0]
#   save_value['distance'] = [float(i) for i in save_value['distance']]
#  save_value.distance.loc[
#     save_value['personID'] ==
#    pID] = save_value.distance[save_value['personID'] == pID] / frameCount

#print(save_value)

#save values file
save_value.to_csv(save_file_name, sep=',', encoding='utf-8')

save_value = save_value.pivot(index='personID',
                              columns='keypoint',
                              values='distance')

for col in save_value.columns:
    try:
        ch = "".join(KEYPOINT_INDEXES[col].split("_"))
    except:
        pass
    try:
        save_value.rename(columns={col: ch}, inplace=True)
    except:
        pass


#Divide by maximum value
def normalize(save_value):
    for col in save_value.columns:
        #for val in save_value[col]:
        print('{} , {}'.format(col, np.max(save_value[col])))
        save_value[col] = save_value[col].astype(float)
        save_value[col] = save_value[col] / np.max(save_value[col])
    return save_value


frameCount = list()
#print(save_value.head())

#Loop to add the frame counts as a new column
for pID in save_value.index:
    frameCount.append(len(pd.read_csv(str(pID) + '.csv')))
#Save distances file
save_value['frameCount'] = frameCount

save_value.to_csv(save_distance, sep=',', encoding='utf-8')
