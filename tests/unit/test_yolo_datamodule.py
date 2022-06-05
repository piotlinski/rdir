import pytest
import torch

from src.datamodules.yolo import YOLODataModule


@pytest.mark.parametrize("batch_size", [32, 128])
@pytest.mark.parametrize(
    "data_dir, config_path",
    [
        ("data/mnist", "data/yolo/results/mnist-yolov4-macho-kangaroo.cfg"),
        ("data/dsprites", "data/yolo/results/dsprites-yolov4-arcane-skink.cfg"),
    ],
)
def test_yolo_datamodule(data_dir, config_path, batch_size):
    datamodule = YOLODataModule(data_dir=data_dir, config_path=config_path, batch_size=batch_size)
    datamodule.prepare_data()

    assert not datamodule.train_dataset and not datamodule.val_dataset

    datamodule.setup()

    assert datamodule.train_dataset and datamodule.val_dataset

    assert datamodule.train_dataloader()
    assert datamodule.val_dataloader()

    batch = next(iter(datamodule.train_dataloader()))
    image, boxes = batch

    assert len(image) == batch_size
    assert len(boxes) == batch_size
    assert image.dtype == torch.float32
    assert boxes.dtype == torch.float32
