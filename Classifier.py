import os
import shutil
import cv2
import tensorflow as tf

from torchreid.utils import FeatureExtractor
from tensorflow.keras import layers
from keras.models import Sequential

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

    model = Sequential()
    model.add(layers.Dense(120, input_dim=512, activation='relu'))
    model.add(layers.Dense(30, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    #model = tf.keras.Model(inputs=input1, outputs=x3)
    optimizer = tf.keras.optimizers.Adam()

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['sparse_categorical_accuracy'])
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

    def initExtractor(self):
        extractor = FeatureExtractor(
            model_name='osnet_x1_0',
            model_path=
            "/home/redev/.cache/torch/checkpoints/osnet_x1_0_imagenet.pth",
            device='cuda')
        return extractor


# create new folder if not exists
def check_new_path(args):
    if not os.path.isdir(args.newPATH):
        os.mkdir(args.newPATH)


# split the data into train, test, val
def file_transfer(args, data, splits):
    # Move all files to the folder
    breakDown = sum(splits)

    for root, directories, files in os.walk(args.PATH):
        print(root)
        try:
            folderNo = root.split('/')[1]
        except:
            continue
        files = files[:breakDown]
        allFiles = list()
        for file in files:
            src = os.path.join(root, file)
            newFileName = folderNo + '_' + file
            dest = os.path.join(args.newPATH, newFileName)
            allFiles.append(dest)
            shutil.copy(src, dest)
        print(len(allFiles))
        train = allFiles[:120]
        val = allFiles[120:140]
        test = allFiles[140:160]
        data["train"]["files"].extend(train)
        data["val"]["files"].extend(val)
        data["test"]["files"].extend(test)

    return data


def get_img(fileName):
    image = cv2.imread(fileName)
    return (image)


# Run torchreid model on all images
def populate_data(data, args):
    for dataset, properties in data.items():
        #      print(dataset)
        #       print(properties)
        data[dataset]['labels'] = [
            name.split('/')[1][0] for name in data[dataset]['files']
        ]
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

        data[dataset]['labels'] = np.array(
            [int(l) for l in data[dataset]['labels']])

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
    trainLab = np.array(data['train']['labels'])
    valFeat = np.array(data['val']['features'])
    valLab = np.array(data['val']['labels'])

    print(trainFeat)
    print(trainLab)

    model = build_model()

    history = model.fit(x=trainFeat,
                        y=trainLab,
                        epochs=2,
                        verbose=3,
                        validation_split=0.2)
    #                        batch_Size=50,
    #       validation_data=(valFeat, valLab))
    #
    res = model.evaluate(data['test']['features'], data['test']['labels'])

    print(history)
    print(res)
