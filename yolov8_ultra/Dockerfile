FROM ultralytics/ultralytics:latest

#COPY . /usr/src/ultralytics
WORKDIR /app
#/usr/src/ultralytics

RUN apt-get update
RUN pip install opencv-contrib-python-headless
RUN pip install opencv-python-headless
RUN pip install deepface
RUN pip install matplotlib
#WORKDIR /usr/src/ultralytics/yolov8_scripts
CMD [ "python", "main.py" ]

#docker build -t testultra .
#docker run  -v /mnt/g/sam_box/:/sam_box -v /home/rdverse/Dev/previousProjects/Responsiveness/yolov8_scripts/:/usr/src/ultralytics/ -itd testultra