import os
import shutil


def is_video(v):

    if v[-3:] in ['MTS', 'MOV', 'm4v']:
        return True
    else:
        return False


if not os.path.isdir('Results'):
    os.mkdir('Results')

videos = os.listdir('box_data')

videos = [v for v in videos if is_video(v)]
print(videos)

for video in videos:
    #Run Step 1 of the analysis
    os.system(
        'python video_demo.py --initialize True --video_input {} --video_output False'
        .format(video))

    #Run Step 2 of the analysis
    os.system('python kmeans.py')

    #Move the results to a folder called results that has a folder with the video name
    outPATH = os.path.join('Results', video)
    os.mkdir(outPATH)

    shutil.move('kmeansPATH', outPATH)
