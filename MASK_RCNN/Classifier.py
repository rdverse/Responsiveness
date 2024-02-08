import os
import shutil
from tkinter import ON
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from torchreid.utils import FeatureExtractor
from tensorflow.keras import layers
from keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder

import torch
import numpy as np
#import pandas as pd

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim


def build_model():
    # input1 = layers.Input(shape=(512, 1))
    # x1 = layers.Dense(120, activation='relu')(input1)
    # x2 = layers.Dense(30, activation='relu')(x1)
    # x3 = layers.Dense(8, activation='softmax')(x2)

    Input = layers.Input(512)
    model = layers.Dense(120, activation='relu')(Input)
    model = layers.Dense(30, activation='relu')(model)
    Output = layers.Dense(4, activation='softmax')(model)
    model = tf.keras.Model(inputs=Input, outputs=Output)

    #model = tf.keras.Model(inputs=input1, outputs=x3)
    optimizer = tf.keras.optimizers.Adam()

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['categorical_accuracy'])
    return model


data = {
    "train": {
        "files": [],
        "labels": [],
        "features": []
    },
    "val": {
        "files": [],
        "labels": [],
        "features": []
    },
    "test": {
        "files": [],
        "labels": [],
        "features": []
    }
}


class globals():
    #Collect all images

    def __init__(self):
        self.PATH = 'chrisPP'
        self.newPATH = 'classChrisPP'
        self.extractor = self.initExtractor()
        self.ohe = OneHotEncoder()

    def initExtractor(self):
        extractor = FeatureExtractor(
            model_name='osnet_x1_0',
            model_path=
            "cache/osnet_x1_0_imagenet.pth",
            device='cuda')
        return extractor
    
    def fit_ohe(self, allLabels):
        self.ohe.fit(np.array(allLabels).reshape(-1,1))

# create new folder if not exists
def check_new_path(args):
    if not os.path.isdir(args.newPATH):
        os.mkdir(args.newPATH)

# split the data into train, test, val
    """
    file_transfer

    Description : Creates a new folder, copies all images there with the corresponding id nos
    
    Args: 
        1) args : contains PATH and newPATH (globals object)
        2) data : global dict to store all values
        3) splits : Depreciated -> pass null
    """

def file_transfer(args, data, splits):
    allFiles = list()
    allLabels = list()
    for root, directories, files in os.walk(args.PATH):
        print(root)
        try:
            folderNo = root.split('/')[1]
        except:
            continue

        for file in files:
            src = os.path.join(root, file)
            newFileName = folderNo + '_' + file
            dest = os.path.join(args.newPATH, newFileName)
            allFiles.append(dest)
            allLabels.append(folderNo)
            shutil.copy(src, dest)
        print(len(allFiles))
        # train = allFiles[:120]
        # val = allFiles[120:140]
        # test = allFiles[140:160]
    trainIndices, testValIndices,a,b = train_test_split(np.arange(len(allFiles)), 
                            np.arange(len(allFiles)), 
                            test_size=0.3, 
                            shuffle = True, random_state = 27)
                    
    testIndices, valIndices,a,b = train_test_split(testValIndices, 
                    testValIndices, 
                    test_size=0.5, 
                    shuffle = True, random_state = 27)
    allFiles=np.array(allFiles)
    data["train"]["files"] = allFiles[[trainIndices]]
    data["val"]["files"] = allFiles[[valIndices]]
    data["test"]["files"] = allFiles[[testIndices]]

    args.fit_ohe(allLabels)
        # data["train"]["files"].extend(train)
        # data["val"]["files"].extend(val)
        # data["test"]["files"].extend(test)

    return data


def get_img(fileName):
    image = cv2.imread(fileName)
    return (image)


# Run torchreid model on all images
def populate_data(data, args):
    ohe = OneHotEncoder()
    for dataset, properties in data.items():
        #      print(dataset)
        #       print(properties)
        print([
            name.split('/')[1].split('_')[0] for name in data[dataset]['files']
        ])
        data[dataset]['labels'] = args.ohe.transform(np.array([
            name.split('/')[1].split('_')[0] for name in data[dataset]['files']
        ]).reshape(-1,1))
        data[dataset]['features'] = [
            args.extractor(get_img(name)) for name in data[dataset]['files']
        ]

        data[dataset]['features'] = [
            feat.cpu().detach().numpy()  #.reshape(-1, 1)
            for feat in data[dataset]['features']
        ]

        shape = len(data[dataset]['features'])
        print(shape)

        data[dataset]['features'] = np.array(
            data[dataset]['features']).reshape(shape, 512)

        # data[dataset]['labels'] = np.array(
        #     [int(l) for l in data[dataset]['labels']])

    allLabels = data['train']['labels']
    allLabels = data['test']['labels']
    allLabels = data['val']['labels']

    return data

# Build a small dense neural network

# Train the model

# Show results

if __name__ == '__main__':
    args = globals()
    check_new_path(args)
    splits = [120, 20, 20]
    data = file_transfer(args, data, splits)
    data = populate_data(data, args)

    trainFeat = np.array(data['train']['features'])
    trainLab = data['train']['labels'].toarray()
    valFeat = np.array(data['val']['features'])
    valLab = data['val']['labels'].toarray()
    testFeat = np.array(data['test']['features'])
    testLab = data['test']['labels'].toarray()

    print(trainFeat)
    print(trainLab)

    model = build_model()

    history = model.fit(x=trainFeat,
                        y=trainLab,
                        validation_data = [valFeat, valLab],
                        epochs=2,
                        verbose=3)
    #                        batch_Size=50,
    #       validation_data=(valFeat, valLab))
    #
    res = model.evaluate(testFeat, testLab)

    print(history)
    print(res)