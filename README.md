## Use the maskrcnn for the humanpose keypoints detection in a group of people


# Requirements
cuda=10.2
python=3.6

# Setting up the directory

1. A folder named box_data containing the videos that you need to process.
2. MaskRCNN weights - can be downloaded from -
3. 


# Environment Setup

1) conda create -n Respy python=3.6

2) pip install pytorch torchvision

3) python -m pip install git+https://github.com/KaiyangZhou/deep-person-reid

4) conda env update -n environment.yml


# Issues

5) If you see a tensorboard error - switch to tensorboard 1.15
pip install tensorboard==1.15

6) Issues with installing deep-person-reid
a. Check pytorch version
b. Check with deep-person-reid repo


# Analysis procedure:

1) Get clippings of the people in the video. (All the masked images will be populated in a new directory groupPATH)
python video_demo.py --initialize True --video_input <video-file-name (located in box_data directory)> --video_output False 

2) Clustering the images in groupPATH
python kmeans.py --n <approximate number of people in the video>
(This will create a new directory kmeansPATH)

3)Manual dataset creation
\n Now create new directories in the root ds/chrisPP/ with some images of each person with a random integer number preferably 0,1,2,3...

For example, if there are three people, there should be 3 folders.
ds/chrisPP/0, ds/chrisPP/1, ds/chrisPP/2. 

4) Collect and save the keypoints for each person.
python video_demo.py --initialize False --video_input <video-file-name (located in box_data directory)> --video_output True

5) Get Responsiveness score for each keypoint for each person.
python distance_calc.py


# Credits
1. 
2. 
3. 
4. 
5. 
