import cv2
import time
import os
import numpy as np
import pandas as pd
PATHdir = ['chrisPP/angles/1a.csv','chrisPP/angles/2a.csv','chrisPP/angles/3a.csv'] 

vidFile = os.path.join('huma.avi')

cap = cv2.VideoCapture(vidFile)
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (frameWidth, frameHeight)
codec = cv2.VideoWriter_fourcc(*'DIVX')
fps = int(cap.get(cv2.CAP_PROP_FPS))

# change these two to alter the fps of read and write
frame_rate_divider = 1  # 1  fps
multiplier = 1 / fps  #1/fps 1
frame_rate_divider = int(fps * multiplier)



output = cv2.VideoWriter('humaDebug.avi', codec, int(1 / multiplier), size)

# read the video
cnt=0
inp = []

# org
org = (00, 185)

# fontScale
fontScale = 1

# Red color in BGR
color = (0, 0, 255)

# Line thickness of 2 px
thickness = 2
font = cv2.FONT_HERSHEY_SIMPLEX

df1 = pd.read_csv(PATHdir[0])

df2 = pd.read_csv(PATHdir[1])

df3 = pd.read_csv(PATHdir[2])

while (cap.isOpened()):
    stime = time.time()
    #try:
    ret, frame = cap.read()
    image_np = np.array([frame.copy()])
    image_np = frame
    # print("image_Np shape")
    # print(image_np.shape)
    #except:
    #    continue
    org1 = []
    org2  = []
    org3 = []


    if ret:
        print(frame)

        cnt+=1
        try:
            text1 = str(int(df1.iloc[cnt]['theta']))
            y = 1080-int(df1.iloc[cnt]['nose_y'])
            x = 1920-int(df1.iloc[cnt]['nose_x'])
            org1 = (x,y)
            
            
        except:
            text1=""
            print("no text 1")
 
        try:
            text2 = str(int(df2.iloc[cnt]['theta']))
            y = 1080-int(df2.iloc[cnt]['nose_y'])
            x = 1920-int(df2.iloc[cnt]['nose_x'])
            org2 = (x,y )
        except:
            text2=""
            print("no text 2")
 
        try:
            text3 = str(int(df1.iloc[cnt]['theta']))
            y = 1080-int(df3.iloc[cnt]['nose_y'])
            x = 1920-int(df3.iloc[cnt]['nose_x'])
            org3 = (x,y)
        except:
            text3=""
            print("no text 3")
        print(org1)
        image_np = cv2.putText(image_np, text1, org1, font, fontScale, (0,0,255), thickness, cv2.LINE_AA, False)

        image_np = cv2.putText(image_np, text2, org2, font, fontScale, (0,255,0), thickness, cv2.LINE_AA, False)

        image_np = cv2.putText(image_np, text3, org3, font, fontScale, (255,0,0), thickness, cv2.LINE_AA, False)

        output.write(image_np)
        # font
    

   
# # Using cv2.putText() method
# image = cv2.putText(image, text, org, font, fontScale, 
#                  color, thickness, cv2.LINE_AA, False)