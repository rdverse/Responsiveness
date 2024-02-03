from mmpose.apis import MMPoseInferencer

img_path = '../tests/data/coco/000000000785.jpg'   
img_path = '/mnt/d/box_data/test.mp4'

inferencer = MMPoseInferencer('human')

# MMPoseInferencer
result_generator = inferencer(img_path, show=True, return_vis=True)
result = next(result_generator)
