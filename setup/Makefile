SHELL := /bin/bash


# build-yolov8:
# 	@echo "Building YOLOv8"
# 	docker build -t yolov8_ultra:latest -f Dockerfile . 

launch-yolov8:
	@echo "Launching YOLOv8"
	docker compose -f docker-compose-yolov8.yaml -p yu2 up

shutdown-yolov8:
	@echo "Shutting down YOLOv8"
	docker compose -f docker-compose-yolov8.yaml -p yu2 down


launch-yolov8-mask:
	@echo "Launching YOLOv8"
	docker compose -f docker-compose-yolov8.yaml -p yum up

shutdown-yolov8-mask:
	@echo "Shutting down YOLOv8"
	docker compose -f docker-compose-yolov8.yaml -p yum down

#docker run -it --rm --gpus all -v $(shell pwd):/workspace -w /workspace --name yolov8_ultra yolov8:latest

# docker build -t yolov8_ultra .
# docker run --gpus=all --shm-size=2gb --name yolov8_ultra_container -v /mnt/g/sam_box:/sam_box -v /home/rdverse/Dev/previousProjects/Responsiveness/yolov8_ultra/scripts:/scripts -itd yolov8_ultra
# docker exec -it yolov8_ultra_container bash
# cd /scripts
# python script_mask.py

# docker container start yolov8_ultra_container
# docker container stop yolov8_ultra_container
# docker container rm yolov8_ultra_container
# docker image rm yolov8_ultra