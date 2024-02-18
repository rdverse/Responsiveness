FROM ultralytics/ultralytics:latest

#COPY . /usr/src/ultralytics
WORKDIR /app
#/usr/src/ultralytics

RUN apt-get update 
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /scripts
#WORKDIR /usr/src/ultralytics/yolov8_script

RUN python emotion.py

ENTRYPOINT ["sleep", "infinity"]

#docker build -t testultra .
#docker run  -v /mnt/g/sam_box/:/sam_box -v /home/rdverse/Dev/previousProjects/Responsiveness/yolov8_scripts/:/usr/src/ultralytics/ -itd testultra