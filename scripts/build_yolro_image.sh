# Build dir docker image
docker build --rm --build-arg WANDB_API_KEY=$WANDB_API_KEY -f docker/rdir.dockerfile -t $RDIR_DOCKER_IMAGE .
