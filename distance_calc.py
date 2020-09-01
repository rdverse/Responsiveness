import re
import pandas as pd
import numpy as np
import os
import glob
import tqdm

######################################3

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

path = 'chrisPP/'
save_file_name = 'saved_values.csv'
save_distance = 'saved_distances.csv'

############################################
#Utils for calculations


def add_comma(match):
    return match.group(0) + ','


def get_euclidean(col_name, col_x, col_y, frames):
    dist = 0
    frameDenominator = 0

    for i in range(len(col_x)):
        #print(frames)
        if i > 0:

            if frames[i] - frames[i - 1] == 1:
                try:
                    dist = dist + np.sqrt((col_x[i] - col_x[i - 1])**2 +
                                          ((col_y[i] - col_y[i - 1])**2))
                    frameDenominator += 1
                except:
                    pass

    dist = dist / frameDenominator
    return (dist, frameDenominator)


#Divide by maximum value
def normalize(save_value):
    for col in save_value.columns:
        #for val in save_value[col]:
        print('{} , {}'.format(col, np.max(save_value[col])))
        save_value[col] = save_value[col].astype(float)
        save_value[col] = save_value[col] / np.max(save_value[col])
    return save_value


########################################

#Change path and create folder
os.chdir(path)

if os.path.isfile(save_file_name):
    os.remove(save_file_name)

if os.path.isfile(save_distance):
    os.remove(save_distance)
######################################

print('Generating file and calculating distance of each keypoint')
save_value = pd.DataFrame(columns=['personID', 'keypoint', 'distance'])

allDist = list()

dataFiles = glob.glob("*.csv")
dataFiles.sort()

for person in dataFiles:

    print('Person : %d' % int(person[0]))
    df = pd.read_csv(person)
    df.sort_values(by='frame', inplace=True)
    col_nos = list(range(17))
    col_mod = [[str(col) + '_x', str(col) + '_y'] for col in col_nos]
    col_mod = np.array(col_mod).flatten()
    calc_df = pd.DataFrame(columns=col_mod)

    #Store the frame numbers in this
    frames = list(df.frame)
    frames = [int(frame) for frame in frames]

    for _, row in tqdm.tqdm(df.iterrows()):

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

    allDistancesMeasured = list()
    col_nos.sort()
    for col in col_nos:
        distance, distancesMeasured = get_euclidean(0,
                                                    calc_df[str(col) + '_x'],
                                                    calc_df[str(col) + '_y'],
                                                    frames)

        save_value.loc[len(save_value)] = np.array([person[0], col, distance])

    allDistancesMeasured.append(distancesMeasured)
    allDist.append(max(allDistancesMeasured))

#save values file
save_value.to_csv(save_file_name, sep=',', encoding='utf-8')

#Reahaping the saved values file
save_value = save_value.pivot(index='personID',
                              columns='keypoint',
                              values='distance')

columns = [str(c) for c in range(17)]
save_value = save_value[columns]
print(save_value.columns)
print(save_value.head())
frameCount = list()

#Loop to add the frame counts as a new column
for pID in save_value.index:
    frameCount.append(len(pd.read_csv(str(pID) + '.csv')))

for col in save_value.columns:
    ch = " ".join(COCO_KEYPOINT_INDEXES[int(col)].split("_"))
    save_value.rename(columns={col: ch}, inplace=True)

#Save distances file
save_value['frameCount'] = frameCount
save_value['Distances Measured'] = allDist
save_value.to_csv(save_distance, sep=',', encoding='utf-8')
