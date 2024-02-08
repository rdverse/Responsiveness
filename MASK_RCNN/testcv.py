import cv2
from skimage.measure import compare_ssim
video_path = "box_data/test3.MTS"
frame_no = 126
frame_no2 = 154

cap = cv2.VideoCapture(video_path) #video_name is the video being called

cap.set(1,frame_no); # Where frame_no is the frame you want
ret, frame1 = cap.read() # Read the frame

cap.set(1,frame_no2); # Where frame_no is the frame you want
ret, frame2 = cap.read() # Read the frame


cropped_frame1 = frame1[200:800, 1000:1900]
cropped_frame2 = frame2[200:800, 1000:1900]

grayA = cv2.cvtColor(cropped_frame1, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(cropped_frame2, cv2.COLOR_BGR2GRAY)

(score,diff) = compare_ssim(grayA, grayB, full  = True)
# cv2.imshow('window_name', cropped_frame1) # show frame on window

# cv2.imshow('window_name2', cropped_frame2) # show frame on window
print("SSIM: {}".format(score))
#If you want to hold the window, until you press exit:
cv2.imwrite("frame.jpg", frame1[200:800, 1000:1900])

cv2.imwrite("frame.jpg", frame1[200:800, 1000:1900])

cv2.imwrite("frame1.jpg", frame2[200:800, 1000:1900])
cv2.waitKey(0) # Wait for a second
cv2.destroyAllWindows()
# print(frame)
# if res:
#     print(frame)
#     cv2.imshow('frame0',frame)
#     cv2.waitKey(0)

