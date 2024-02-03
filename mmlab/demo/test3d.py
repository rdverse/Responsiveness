from mmpose.apis import MMPoseInferencer

img_path = '../tests/data/coco/000000000785.jpg'   
img_path = '/mnt/d/box_data/test.mp4'



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
result_generator = inferencer(img_path, show=True, return_vis=True, use_oks_tracking=True)
#result = next(result_generator)
results = [result for result in result_generator]
