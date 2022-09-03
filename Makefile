ifdef gpus
	GPUS := "device=$(gpus)"
else
	GPUS := all
endif

help: ## Show this help
	@grep -E '^[.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

format:  ## Run pre-commit hooks to format code
	pre-commit run --all-files

build.darknet:  ## Build darknet docker image
	docker build --rm -f docker/darknet.dockerfile -t $(DARKNET_DOCKER_IMAGE) .
	docker run -v $(PWD):/app --name darknet_builder --gpus all $(DARKNET_DOCKER_IMAGE) /app/docker/build_darknet.sh
	docker commit darknet_builder $(DARKNET_DOCKER_IMAGE)
	docker rm darknet_builder

build.rdir:  ## Build dir docker image
	docker build --rm --build-arg WANDB_API_KEY=$(WANDB_API_KEY) -f docker/rdir.dockerfile -t $(RDIR_DOCKER_IMAGE) .

shell.dir:  ## Run shell in dir docker container
	docker run -it --rm --ipc=host -v $(PWD):/workspace -e LOCAL_USER_ID=$(LOCAL_USER_ID) -e LOCAL_GROUP_ID=$(LOCAL_GROUP_ID) --gpus '$(GPUS)' $(RDIR_DOCKER_IMAGE)
