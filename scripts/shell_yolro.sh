# Run shell in dir docker image
docker run -it --rm -v $PWD:/workspace -e LOCAL_USER_ID=$LOCAL_USER_ID -e LOCAL_GROUP_ID=$LOCAL_GROUP_ID --gpus all $RDIR_DOCKER_IMAGE
