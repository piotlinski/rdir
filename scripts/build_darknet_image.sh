#!/bin/bash
# Build darknet docker image
docker build --rm -f docker/darknet.dockerfile -t $DARKNET_DOCKER_IMAGE . && \
docker run -v $PWD:/app --name darknet_builder --gpus all $DARKNET_DOCKER_IMAGE /app/docker/build_darknet.sh && \
docker commit darknet_builder $DARKNET_DOCKER_IMAGE && \
docker rm darknet_builder
