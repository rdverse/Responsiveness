from mmpose.apis import MMPoseInferencer

img_path = '../tests/data/coco/000000000785.jpg'   
#img_path = '/mnt/d/box_data/tests/testmin.mp4'
img_path = '../../box_data/tests/positive.mp4'

import matplotlib.pyplot as plt
POSITION={
    'Nose': 0,
    'Left_Eye': 1,
    'Right_Eye': 2,
    'Left_Ear': 3,
    'Right_Ear': 4,
    'Left_Shoulder': 5,
    'Right_Shoulder': 6,
    'Left_Elbow': 7,
    'Right_Elbow': 8,
    'Left_Wrist': 9,
    'Right_Wrist': 10,
    'Left_Hip': 11,
    'Right_Hip': 12,
    'Left_Knee': 13,
    'Right_Knee': 14,
    'Left_Ankle': 15,
    'Right_Ankle': 16
}

def normalize_keypoints(keypoints ):
    """
    Normalize keypoints to be in the range [0, 1].

    Parameters:
        keypoints (np.ndarray): Keypoints (shape: (17, 3)).
        width (int): Width of the image.
        height (int): Height of the image.

    Returns:
        np.ndarray: Normalized keypoints (shape: (17, 3)).
    """
    import numpy as np
    # copy keypoints
    norm_kps = np.copy(keypoints)
    norm_kps = (norm_kps + 1)*0.5
    return norm_kps

def compute_gaze_direction(left_ear_keypoint, right_ear_keypoint, left_eye_keypoint, right_eye_keypoint):
    """
    Compute gaze direction based on ear and eyes keypoints.

    Parameters:
        left_ear_keypoint (np.ndarray): Keypoint for the left ear (shape: (3,)).
        right_ear_keypoint (np.ndarray): Keypoint for the right ear (shape: (3,)).
        left_eye_keypoint (np.ndarray): Keypoint for the left eye (shape: (3,)).
        right_eye_keypoint (np.ndarray): Keypoint for the right eye (shape: (3,)).

    Returns:
        str: Gaze direction ('Top Right' or 'Other').
    """
    import numpy as np
    left_ear_keypoint = normalize_keypoints(left_ear_keypoint)
    right_ear_keypoint = normalize_keypoints(right_ear_keypoint)
    left_eye_keypoint = normalize_keypoints(left_eye_keypoint)
    right_eye_keypoint = normalize_keypoints(right_eye_keypoint)
    
    # check which quadrant the person is looking at
    mean_ear_keypoint = np.array(left_ear_keypoint + right_ear_keypoint) /2 
    mean_eye_keypoint = np.array(left_eye_keypoint + right_eye_keypoint) /2
    if mean_eye_keypoint[0] > mean_ear_keypoint[0] and mean_eye_keypoint[1] < mean_ear_keypoint[1]:
        return 'Right'
    elif mean_eye_keypoint[0] < mean_ear_keypoint[0] and mean_eye_keypoint[1] < mean_ear_keypoint[1]:
        return 'Left'
    elif mean_eye_keypoint[0] > mean_ear_keypoint[0] and mean_eye_keypoint[1] > mean_ear_keypoint[1]:
        return 'Right'
    elif mean_eye_keypoint[0] < mean_ear_keypoint[0] and mean_eye_keypoint[1] > mean_ear_keypoint[1]:
        return 'Left'
    else:
        return 'Other'


def mpl_to_opencv_image(fig):
    """
    Convert Matplotlib figure to an OpenCV image.
    """
    # Render the figure to a buffer
    buf, (w, h) = fig.canvas.print_to_buffer()
    buf_rgba = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))

    # Convert RGBA to RGB
    buf_rgb = buf_rgba[:, :, :3]

    return buf_rgb 

def compute_movement(prev_kps, curr_kps):
    """
    Compute movement of person based on previous and current keypoints.

    Parameters:
        prev_kps (np.ndarray): Previous keypoints (shape: (17, 3)).
        curr_kps (np.ndarray): Current keypoints (shape: (17, 3)).

    Returns:
        str: Movement of person ('Moving' or 'Stationary').
    """
    import numpy as np
    # compute difference between previous and current keypoints
    diff = np.array(prev_kps) - np.array(curr_kps)
    # compute mean of differences
    mean_diff = np.mean(abs(diff), axis=0)
    # compute magnitude of mean difference
    magnitude = np.sqrt(mean_diff[0] ** 2 + mean_diff[1] ** 2)
    print(magnitude)
    if magnitude > 0.004:
        state = 'Moving'
    else:
        state = 'Stationary'
    return magnitude, state

# build the inferencer with 3d model alias
inferencer = MMPoseInferencer(pose3d="human3d")

# build the inferencer with 3d model config name
inferencer = MMPoseInferencer(pose3d="motionbert_dstformer-ft-243frm_8xb32-120e_h36m")
import os
print(os.listdir("../configs/body_3d_keypoint/motionbert/h36m/"))
print(os.path.isfile("../configs/body_3d_keypoint/motionbert/h36m/motionbert_dstformer-ft-243frm_8xb32-120e_h36m.py"))
# build the inferencer with 3d model config path and checkpoint path/URL
inferencer = MMPoseInferencer(
    pose3d='../configs/body_3d_keypoint/motionbert/h36m/motionbert_dstformer-ft-243frm_8xb32-120e_h36m.py',
    
           pose3d_weights='https://download.openmmlab.com/mmpose/v1/body_3d_keypoint/' \
                   'pose_lift/h36m/motionbert_ft_h36m-d80af323_20230531.pth'
)

# MMPoseInferencer
result_generator = inferencer(img_path, show=False, return_vis=True, use_oks_tracking=True)
#result = next(result_generator)
results = []
import tqdm
for result in tqdm.tqdm(result_generator):
    #print(result)
    results.append(result)

    if len(results) > 400:
        break
    #
    # save the results as a video
    #result.render()
   # break
   
import cv2
cap = cv2.VideoCapture(img_path)
# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS)) // 8 
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create VideoWriter object to save the output video
output_path = "../../box_data/tests/output/output_gaze.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
rolling_movement = []

x_data = []
# Initialize Matplotlib plot
# plt.ion()
# fig, ax = plt.subplots()
# ax.set_title("Movement Over Time")
# ax.set_xlabel("Frame")
# ax.set_ylabel("Movement Score (%)")
# line, = ax.plot([], [], label="Movement Score")
# ax.legend()
# Iterate through each frame in the video
for i,result in tqdm.tqdm(enumerate(results)):
    
    if i==0:
        curr_kps = result['predictions'][0][0]['keypoints']
        prev_kps = curr_kps
    else:
        prev_kps = curr_kps
        curr_kps = result['predictions'][0][0]['keypoints']
        
    
    image_np = result['visualization']
    #frame = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    frame = image_np[0]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = result['predictions'][0][0]
    # Get keypoints for ear and eyes
    left_ear = result['keypoints'][POSITION['Left_Ear']]
    right_ear = result['keypoints'][POSITION['Right_Ear']]
    left_eye = result['keypoints'][POSITION['Left_Eye']]  # Assuming left eye keypoint is at index 1
    right_eye = result['keypoints'][POSITION['Right_Eye']]  # Assuming right eye keypoint is at index 2

    # Compute gaze direction
    gaze_direction = compute_gaze_direction(left_ear, right_ear, left_eye, right_eye)

    # Draw boundary based on gaze direction
    color = (255, 0, 0) if gaze_direction == 'Right' else (0, 165, 255)  # Blue for 'Top Right', Orange for 'Other'
    #cv2.rectangle(frame, (0, 0), (width, height), color, 5)
    gaze_text = 'Gaze Direction: ' + gaze_direction
    cv2.putText(frame, gaze_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 1, cv2.LINE_AA)
    
    movement, state = compute_movement(prev_kps, curr_kps)
    rolling_movement.append(movement)
    import numpy as np
    movement_text = 'Movement: ' + state + ' (' + str(np.mean(rolling_movement)) + ')' 
    movement_text = "Movement : {} , Score : {:.4f}".format(state, np.mean(rolling_movement))
    nframes_text = "Nframes moved : {}".format(np.sum(np.array(rolling_movement) > 0.004))
    cv2.putText(frame, movement_text, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)
    cv2.putText(frame, nframes_text, (100, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)
    # # Update real-time plot
    # x_data.append(i)
    # y_data = rolling_movement
    # line.set_xdata(x_data)
    # line.set_ydata(y_data)
    # ax.relim()
    # ax.autoscale_view()
    
    # Convert Matplotlib plot to an image
    # plot_img = mpl_to_opencv_image(fig)

    # Resize plot image to match video frame dimensions
    # plot_img = cv2.resize(plot_img, (width, int(height / 4)))  # Adjust height as needed

    # # Overlay plot image onto the top of the video frame
    # frame[frame.shape[0]-plot_img.shape[0]:, :] = cv2.addWeighted(frame[frame.shape[0]-plot_img.shape[0]:, :], 0.1, plot_img, 1, 0)

    
    # Write the frame with text to the output video
    out.write(frame)
        # Display the frame with text
    cv2.imshow('Frame', frame)
    if i==0:
        cv2.waitKey(1500)
    cv2.waitKey(900)
    # Update Matplotlib plot
    plt.pause(0.01)
    
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()