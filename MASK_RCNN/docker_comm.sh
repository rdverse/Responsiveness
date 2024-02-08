#!/bin/bash

cd ../../

# # stop all running containers
docker stop $(docker ps -aq)
# remove all containers
docker rm $(docker ps -aq)
# # remove all image with name "depthdoc"
docker rmi $(docker images | grep docdepth3 | awk '{print $3}')

# Build the Docker image with the provided Dockerfile
docker build -f XAI_VISION/depthmodels/Dockerfile . -t docdepth3

# Run the Docker container, mounting specified folders
docker run --gpus all -itd \
    -v /mnt/d/eagle:/eagle \
    -v /home/rdverse/Dev/currentProjects/XAI_VISION:/app/XAI_VISION \
    --name condocdepth3 docdepth3

#  docker run --gpus all -itd \
#     -v /mnt/g/eagle:/eagle \
#     --name depthdoccon depthdoc3
   
# Execute bash inside the running container
docker exec -itd condocdepth3 dir_setup.sh

docker exec -it condocdepth3 bash