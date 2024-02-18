# Instructions to run the docker container

```  
docker build -t yolov8_ultra .
docker run --gpus=all --shm-size=6gb --name yolov8_ultra_container -v /mnt/g/sam_box:/sam_box -v /home/rdverse/Dev/previousProjects/Responsiveness/yolov8_ultra/scripts:/scripts -itd yolov8_ultra
docker exec -it yolov8_ultra_container bash
cd /scripts
python script_mask.py


docker container stop yolov8_ultra_container
docker container rm yolov8_ultra_container
```  

