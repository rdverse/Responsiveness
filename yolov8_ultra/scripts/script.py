from ultralytics import YOLO
import cv2
import numpy as np
from ultralytics import YOLO

from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator, colors

from collections import defaultdict
# Load an official or custom model
model = YOLO('yolov8n.pt')  # Load an official Detect model
model = YOLO('yolov8n-seg.pt')  # Load an official Segment model
#model = YOLO('path/to/best.pt')  # Load a custom trained model
track_history = defaultdict(lambda: [])

model = YOLO('yolov8x-pose-p6.pt')  # Load an official Pose model
names = model.model.names
print(names)

# Open the video file
video_path = "/sam_box/inputs_huma/snippet.mp4"
cap = cv2.VideoCapture(video_path)

cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

result = cv2.VideoWriter("/sam_box/outputs_huma/object_tracking_large.avi",
                       cv2.VideoWriter_fourcc(*'mp4v'),
                       fps,
                       (w, h))

while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model.track(frame.copy(), persist=True, verbose=False)
        boxes = results[0].boxes.xyxy.cpu()

        if results[0].boxes.id is not None:

            # Extract prediction results
            clss = results[0].boxes.cls.cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confs = results[0].boxes.conf.float().cpu().tolist()
            keypoints = results[0].keypoints.cpu()#.tolist()
            print(type(keypoints))
            # Annotator Init
            annotator = Annotator(frame, line_width=2)

            for box, cls, track_id, conf, keypoint in zip(boxes, clss, track_ids, confs, keypoints):
                print(f"class : {cls}")
                if int(cls)==0:
                    annotator.box_label(box, color=colors(int(cls), True), label=f"(confidence:{conf:.2f}):{names[int(cls)]}:{track_id}")
                    annotator.kpts(keypoint.data.squeeze())
                    
                # Store tracking history
                track = track_history[track_id]
                track.append((int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)))
                if len(track) > 30:
                    track.pop(0)

                # Plot tracks
                points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                cv2.circle(frame, (track[-1]), 7, colors(int(cls), True), -1)
                cv2.polylines(frame, [points], isClosed=False, color=colors(int(cls), True), thickness=2)

        result.write(frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

result.release()
cap.release()
cv2.destroyAllWindows()
