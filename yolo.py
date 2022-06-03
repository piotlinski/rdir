import hydra
from omegaconf import DictConfig


@hydra.main(config_path="configs/", config_name="yolo.yaml")
def main(config: DictConfig):
    from src import utils
    from src.darknet_pipeline import yolo

    # Applies optional utilities
    utils.extras(config)

    # train yolov4
    return yolo(config)


if __name__ == "__main__":
    main()
