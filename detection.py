import csv
import os
import torch
import csv
import pandas as pd
import numpy as np
import cv2
import torchreid
from torchreid.utils import FeatureExtractor
from torch.nn import CosineSimilarity
from TS_SS import TS_SS
'''
Class:
--------
Purpose: # A class that stores only feature vectors of the comparision images
--------
Arguments: paths
---------
Parameters: None
----------
Returns: # basically end product is a dictionary
------
'''
print('detections is called')


class compareData():
    '''
    stores compData - a dictionary with feature vectors of all comparision people
    Properties: key - personID, values - list of feature vectors.
    ----------
    '''
    def __init__(self):
        #The only thing we need from here
        self.compData = dict()

        self.extractor = self.initExtractor()
        # Hardcoded output directory
        self.mainPath = 'chrisPP'
        self.mainedPath = 'ds/chrisPP'
        #self.mainedPath = os.path.join('Results/{}/ds/'.format(video_name)

        features = list()
        # Go over comparision images
        for personID in os.listdir(self.mainedPath):
            self.compData[personID] = self.get_features(personID)
            self.dir_setup(personID)
        self.compData = self.convert_to_cpu()

    def convert_to_cpu(self):
        compData = self.compData
        for key, compareFeatureList in self.compData.items():
            compData[key] = [
                k.cpu().detach().numpy() for k in compareFeatureList
            ]
            #compData[key] = compareFeatureList.cpu().detach().numpy()
        return (compData)

    ''' Initialize and return the torchreid extractor'''

    def initExtractor(self):
        extractor = FeatureExtractor(
            model_name='osnet_x1_0',
            model_path= 'cache/osnet_x1_0_imagenet.pth',
            device='cuda')  
        return extractor

    ''' 
        Arguments: personID
        ----------
        Returns: feature vectors list of each person
        --------
'''

    def get_features(self, personID):
        features = list()
        personPath = os.path.join(self.mainedPath, personID)
        for imName in os.listdir(personPath):
            imagePath = os.path.join(personPath, imName)
            image = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)
            featureVector = self.get_feature(image)
            features.append(featureVector)
        return features

    ''' 
        Arguments: image
        ----------  
        Returns: feature vectors list of each person
        --------
'''

    def get_feature(self, image):
        #print("extractor image")
        #print(image.shape)
        return (self.extractor(image))

    '''
    Create a folder and csv for each of the people
'''

    def dir_setup(self, personID):

        # create folder
        if not os.path.isdir(self.mainPath):
            os.mkdir(self.mainPath)

        personPath = os.path.join(self.mainPath, personID)

        if not os.path.isdir(personPath):
            os.mkdir(personPath)

        #Create the csv file
        csvHead = [
            'personID', 'frame', 'feature_vectors', 'pose_preds', 'score'
        ]

        fileName = os.path.join(self.mainPath, personID + '.csv')
        with open(fileName, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(csvHead)
            csvwriter.writerow([0,0,0,0,0])



# basically end product is a dictionary
# basically end product is a dictionary
'''
Person nodes to store values of each person data
'''


class Person():
    def __init__(self, data, personID):
        self.image = data['images'][personID]
        self.keypoints = data['keypoints'][personID]
        self.keypointScore = data['scores'][personID]
        self.featureVector = None
        self.bestMatch = None
        self.bestScore = None

    def set_feature_vector(self, k):
        self.featureVector = k.cpu().detach().numpy()


'''
Main class to classify people and assign labels

'''


class DetectionSetupMode(compareData):
    def __init__(self):
        # This sets up the comparision data
        self.compareObject = compareData()
        self.compData = self.compareObject.compData

        # count of the frame
        self.cnt = 0
        self.cos = CosineSimilarity(dim=1, eps=1e-08)
        #self.cos = TS_SS()

        # collection of detections - list of person nodes
        personDetections = list()

        # Hardcoded output directory
        self.mainPath = 'chrisPP'
        self.mainedPath = 'ds/chrisPP'

    def classifyDetections(self, data, cnt):
        self.cnt = cnt
        for personID, item in enumerate(data['images']):
            # First initialize this and then figure out the three unknowns
            person = Person(data, personID)

            #Get feature vector for this image and convert_to_cpu
            person.set_feature_vector(
                self.compareObject.get_feature(person.image))

            bestMatch, bestScore = self.who_is_this(person)

            if bestMatch:
                # Save image
                imgFoldPath = os.path.join(self.mainPath, str(bestMatch))
                imgPath = os.path.join(imgFoldPath, str(self.cnt) + '.jpg')
                csvPath = os.path.join(self.mainPath, str(bestMatch) + '.csv')

                if os.path.exists(imgPath) and len(
                        os.listdir(imgFoldPath)) > 3:
                    #print(imgPath)
                    df = pd.read_csv(csvPath, index_col=0)
                    #print(df)
                    if df.iloc[-1]['score'] > bestScore:
                        continue
                    else:
                        df = df[:-1]
                        df.reset_index(inplace=True, drop=True)
                        df.to_csv(csvPath)

                cv2.imwrite(imgPath, person.image)

                # Log results in a csv file
                el1 = bestMatch
                el2 = self.cnt
                el3 = person.featureVector
                el4 = person.keypoints
                el5 = bestScore
                csvRow = [el1, el2, el3, el4, el5]
                with open(csvPath, 'a', newline='') as csvFile:
                    csvWriter = csv.writer(csvFile)
                    csvWriter.writerow(csvRow)

    '''
    Function: who_is_this
    -----
    Purpose: get the match and score
    --------
    Arguments: person object
    ---------
    Parameters: 
    ----------
    Returns: tuple. (bestMatch,bestScore)
    -------

    '''

    def who_is_this(self, person):
        compareList = list()
        scoreList = list()
        for key, compareFeatureList in self.compareObject.compData.items():
            scores = list()
            for cFeat in compareFeatureList:
                #score = self.cos(cFeat, person.featureVector)
                #scores.append(score[0][0])

                score = self.cos(torch.from_numpy(cFeat),
                                 torch.from_numpy(
                                     person.featureVector)).cpu().item()
                scores.append(score)

            #Identify the minimum score of each compared person
            scoreList.append(max(scores))
            compareList.append(key)

        # best is the least score among all
        bestIndex = scoreList.index(max(scoreList))
        bestScore = scoreList[bestIndex]
        bestMatch = compareList[bestIndex]

        if bestScore > 0.15:
            return (bestMatch, bestScore)

        else:
            return (False, False)
