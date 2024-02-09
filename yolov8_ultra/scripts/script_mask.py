#edit from terminal
from ultralytics import YOLO
import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import numpy as np
from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator, colors

from collections import defaultdict
# Load an official or custom model
model = YOLO('yolov8n.pt')  # Load an official Detect model
model = YOLO('yolov8n-seg.pt')  # Load an official Segment model
#model = YOLO('path/to/best.pt')  # Load a custom trained model
track_history = defaultdict(lambda: [])
model = YOLO('yolov8x-pose-p6.pt')  # Load an official Pose model
model = YOLO('yolov8n-seg.pt')  # Load an official Segment model
model = YOLO('yolov8x-seg.pt')  # Load an official Segment model

names = model.model.names
print(names)

# function recieves frameno, masks, boxes, track_ids, confs, clss, image-frame
# it should create a folder if not exists with the name of track_id and save the mask image that is cut from the frame
# the saved image should be named with the frame number
# a separate folder to be created to store the json file with the track history
def save_masks(frame_no, masks, boxes, track_ids, confs, clss, frame):
    for mask, box, track_id, conf,cls in zip(masks, boxes, track_ids, confs,clss):
            if int(cls)!=0:
                continue
            print(f"mask shape {mask.shape}")
            print(f"frame shape {frame.shape}")
            #mask = np.array([cv2.resize(mask, (w,h))])
            frame_masked = frame.copy()*mask[... , np.newaxis]
            cv2.imwrite("/sam_box/outputs_huma/frame_masked.png", frame_masked)
            print(f"frame_masked shape {frame_masked.shape}")
            # mask = mask[0]
            # mask = mask.astype(np.uint8)
            # mask = mask*255
            # mask = mask.astype(np.uint8)
            
            x1, y1, x2, y2 = box.cpu().int().tolist()
            print(f"box coordinates {x1, y1, x2, y2}")
            frame_masked = frame_masked[y1:y2, x1:x2,]
            if not os.path.exists(f"/sam_box/outputs_huma/masks/{track_id}"):
                os.makedirs(f"/sam_box/outputs_huma/masks/{track_id}")
            cv2.imwrite(f"/sam_box/outputs_huma/masks/{track_id}/{frame_no}.png", frame_masked)
            # save the track history in a json file
            # save the mask image in the folder with the name of track_id
            # the name of the mask image should be the frame number
            # the mask image should be cut from the frame using the box coordinates
            # the mask image should be resized to the original frame size
            # the mask image should be saved in the folder with the name of track_id
            # the mask image should be saved with the name of the frame number





# Open the video file
video_path = "/sam_box/inputs_huma/PP_SI_DAY11.MTS"
cap = cv2.VideoCapture(video_path)

assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

result = cv2.VideoWriter("/sam_box/outputs_huma/object_tracking_mask3_large.avi",
                       cv2.VideoWriter_fourcc(*'mp4v'),
                       fps//29,
                       (w, h))

color_list = colors.pose_palette
frame_count = 0
while cap.isOpened():
    success, frame = cap.read()
    frame_stash = frame.copy()
    frame_count+=1
    print(frame_count)
    if frame_count%29!=0:
        continue
    if success:
        results = model.track(frame.copy(), persist=True, verbose=False)
        boxes = results[0].boxes.xyxy.cpu()

        if results[0].boxes.id is not None:

            # Extract prediction results
            clss = results[0].boxes.cls.cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confs = results[0].boxes.conf.float().cpu().tolist()
            masks = results[0].masks.data.cpu()
            # Annotator Init
            annotator = Annotator(frame, line_width=2)
            n,mw,mh = masks.shape
            im_gpu = frame.copy()
            #im_gpu = cv2.resize(im_gpu,(mh,mw))
            im_gpu = np.rollaxis(im_gpu,2,0)
            cv2.imwrite("/sam_box/outputs_huma/frame.png", im_gpu)
            im_gpu = torch.tensor(im_gpu)
            masks = masks.detach().cpu().numpy()
            masks = np.array([cv2.resize(mask, (w,h)) for mask in masks.copy()])
            #print(f"shaep of input mask 2 {np.sum(masks, axis=2)} and  0 {np.sum(masks, axis=0)}")
            #cv2.imwrite("/sam_box/outputs_huma/masks.png",np.sum(masks,axis=0) )
            masks = torch.tensor(masks)
            while len(color_list)<=np.max(track_ids):
                color_list = np.concatenate((color_list, colors.pose_palette))
            
            annotator.masks(masks, color_list[track_ids], im_gpu/255)

            save_masks(frame_count, masks.cpu().numpy(), boxes, track_ids, confs, clss, frame_stash)
            for box, cls, track_id, conf, mask in zip(boxes, clss, track_ids, confs, masks.cpu().numpy()):
                #print(f"class : {cls}")
                if int(cls)==0:
                    annotator.box_label(box, color=colors(int(cls), True), label=f"(confidence:{conf:.2f}):{names[int(cls)]}:{track_id}")
                    
                # Store tracking history
                track = track_history[track_id]
                track.append((int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)))
                if len(track) > 30:
                    track.pop(0)

                # Plot tracks
                points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                cv2.circle(frame, (track[-1]), 7, colors(int(cls), True), -1)
                cv2.polylines(frame, [points], isClosed=False, color=colors(int(cls), True), thickness=2)
        cv2.imwrite("/sam_box/outputs_huma/final_frame.png",frame)
        result.write(frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

result.release()
cap.release()
cv2.destroyAllWindows()
