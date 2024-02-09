import os
import cv2
from deepface import DeepFace
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import logging

# global variables
# Video input path
dir_path = "/sam_box/outputs_huma/masks"
output_dir = f"/sam_box/outputs_huma/masks_video/"
width = 1280#xmax 
height = 720#ymax 
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


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


def get_xy_max_values(input_dir_path, faces):
    """
    Get the minimum and maximum values from the running average emotions.
    """
    sizes = []

    # Get video properties
    for face in tqdm.tqdm(faces[10:-10]):
        read_img = cv2.imread(os.path.join(input_dir_path, face)) 
        sizes.append(read_img.shape)

    sizes = np.array(sizes)
    xmax, ymax = np.max(sizes[:,0]) , np.max(sizes[:,1])

    return xmax, ymax

# Initialize Matplotlib plot
def get_plot():
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title("Happy Score Over Time")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Happy Score (%)")
    ax.set_ylim([0, 100])
    line, = ax.plot([], [], label="Happy Score", color="black", linewidth=4)
    ax.legend()
    return fig, ax, line

person_id = "10"

input_dir_path = os.path.join(dir_path, person_id)    
faces  = os.listdir(input_dir_path)
faces.sort(key=lambda x : int(x.strip(".png")))

xmax, ymax = get_xy_max_values(input_dir_path, faces)

# Create VideoWriter object to save the output video
output_path = os.path.join(output_dir, f"{person_id}.avi") 
fps = 1#int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (1280, 720))

# Initialize variables for running average and happy scores
running_average_emotion = {"angry": 0, "disgust": 0, "fear": 0, "happy": 0, "sad": 0, "surprise": 0, "neutral": 0}
frame_count = 0
happy_scores = []

fig, ax, line = get_plot()

# Initialize variables for real-time plotting
x_data = []
y_data = []
#sizes=[]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.info(f"dir_path {input_dir_path}")
    
        # pad all images to have same size as the largest image
    for face in faces[10:-10]:
        img_canvas = np.full((xmax, ymax, 3), 255, dtype=np.uint8)
        read_img = cv2.imread(os.path.join(input_dir_path, face)) 
        # pad image to have same size as the largest image
        image_center = [int(read_img.shape[0]/2), int(read_img.shape[1]/2)] 
        img_canvas[0:read_img.shape[0], 0:read_img.shape[1]] = read_img
        frame = img_canvas
        obj = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        obj = obj[0]
        instantaneous_emotion = obj["emotion"]
        happy_scores.append(instantaneous_emotion["happy"])

        # Update running average emotions
        frame_count += 1
        for emotion in running_average_emotion:
            running_average_emotion[emotion] += (instantaneous_emotion[emotion] - running_average_emotion[emotion]) / frame_count

        # Define the target resolution
        # Format text to display on the video
        text_instantaneous = f"Instantaneous Emotion: {obj['dominant_emotion']}"
        text_average = f"Running Average Emotion: {max(running_average_emotion, key=running_average_emotion.get)}"
        text_happy_score = f"Average Happy Score: {np.mean(happy_scores):.2f}/100"

        # Overlay text on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_thickness = 1
        text_color = (255, 255, 255)
        
        text_position_instantaneous = (30, int(height / 4) + 70)
        text_position_average = (30, int(height / 4) + 90)
        text_position_happy_score = (30, int(height / 4) + 110)
         
        frame = cv2.resize(frame, (width, height))
        cv2.putText(frame, text_instantaneous, text_position_instantaneous, font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        cv2.putText(frame, text_average, text_position_average, font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        cv2.putText(frame, text_happy_score,text_position_happy_score , font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Frame: {frame_count}", (30, 30), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        
        # Update real-time plot
        x_data.append(frame_count)
        y_data = happy_scores
        line.set_xdata(x_data)
        line.set_ydata(y_data)
        #ax.set_ylim([0, 100])
        ax.relim()
        ax.autoscale_view()

        # Convert Matplotlib plot to an image
        plot_img = mpl_to_opencv_image(fig)

        #Resize plot image to match video frame dimensions
        plot_img = cv2.resize(plot_img, (width, int(height / 4)))  # Adjust height as needed

        #Overlay plot image onto the top of the video frame
        frame[frame.shape[0]-plot_img.shape[0]:, :] = cv2.addWeighted(frame[frame.shape[0]-plot_img.shape[0]:, :], 0, plot_img, 1, 0)
        
        
        out.write(frame)
     
    out.release()
    
    # plot the data with appropriate labels
    plt.plot(x_data, y_data, label="Happy Score", color="black", linewidth=4)
    plt.title("Happy Score Over Time")
    plt.xlabel("Frame")
    plt.ylabel("Happy Score")
    plt.savefig(f"/sam_box/outputs_huma/masks_video/{person_id}.png")