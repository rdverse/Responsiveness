from sklearn.cluster import KMeans
import numpy as np
import os
import glob
import shutil
from torchreid.utils import FeatureExtractor
import cv2
import tqdm

PATH = 'chrisPP'
# Populate all files in a folder
# Check for kmeans folder
kmeansPATH = 'kmeansPATH'
groupPATH = 'groupPATH'

# flag to determine weather to populate or not
populate = False
if not os.path.isdir(groupPATH):
    os.mkdir(groupPATH)
    populate = True

if not os.path.isdir(kmeansPATH):
    os.mkdir(kmeansPATH)

print('Finished creating main folders...')
extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    model_path="/home/redev/.cache/torch/checkpoints/osnet_x1_0_imagenet.pth",
    device='cuda')

if populate:
    folders = os.listdir(PATH)
    folders = [f for f in folders if f[-1] != 'v']
    #This will put everything in the groupPATH
    print('Populating all files into the group folder')
    for folder in tqdm.tqdm(folders):
        folderPATH = os.path.join(PATH, folder)
        for file in os.listdir(folderPATH):
            srcFile = os.path.join(PATH, folder, file)
            destFile = os.path.join(groupPATH, folder + '_' + file)
            shutil.copy(srcFile, destFile)

# Find feature vectors of all the images
print('Finding the feature vectors of all the images')
labels = [image[0] for image in glob.glob(groupPATH)]

features = list()

images = glob.glob(os.path.join(groupPATH, '*.*'))
for image in tqdm.tqdm(images):
    img = cv2.imread(image)
    feature = extractor(img)
    features.append(list(feature.detach().cpu().numpy())[0])
#print(features)
print('Training the model')
# Run kmeans clustering algorithm
#print(np.array(features).shape)
model = KMeans(n_clusters=10, max_iter=5000, n_jobs=-1,
               verbose=1).fit(features)

# Reassign labels for all images after training the model
predLabels = model.labels_

print(
    'Using the model to reassign the labels and move the images appropriately')
for image, label in list(zip(images, predLabels)):
    label = str(label)
    newPATH = os.path.join(kmeansPATH, label)
    if not os.path.isdir(newPATH):
        os.mkdir(newPATH)
    imageName = image.split('/')[-1].split('_')[-1]
    newPATH = os.path.join(newPATH, imageName)
    shutil.copy(image, newPATH)

shutil.rmtree(groupPATH, ignore_errors=True)
