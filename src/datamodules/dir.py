"""DIR sequential DataModule."""
import shutil
import tarfile
from itertools import groupby
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from src.datamodules.yolo import YOLODataset
from src.utils import get_logger

logger = get_logger(__name__)


class RDIRDataset(YOLODataset):
    """Sequential dataset for training DIR."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.files = self._split_sequences(self.files)

    @staticmethod
    def _split_sequences(files: List[str]) -> List[List[str]]:
        """Create list of sequences of images."""

        def keyfunc(x: str) -> str:
            return "/".join(x.split("/")[:-1])

        files_sorted = sorted(files, key=keyfunc)
        return [list(g) for k, g in groupby(files_sorted, key=keyfunc)]

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get sequence of images from the dataset."""
        images = []
        boxes = []
        for i, img_path in enumerate(self.files[idx]):
            if not self.is_train:
                img, xywh = self.get_original(img_path)
                img = cv2.resize(img, (self.width, self.height))
            else:
                img, x1y1x2y2 = self.get_augmented(img_path, reuse=(i != 0))
                xywh = self.x1y1x2y2_to_xywh(x1y1x2y2, (self.width, self.height))
            seq_boxes = np.zeros([self.max_boxes, 5])
            if xywh.shape[0] > 0:
                seq_boxes[: min(xywh.shape[0], self.max_boxes)] = xywh[
                    : min(xywh.shape[0], self.max_boxes)
                ]
            images.append(img.astype(np.float32))
            boxes.append(seq_boxes.astype(np.float32))
        return np.stack(images, axis=0), np.stack(boxes, axis=0)


def seq_collate(batch):
    """Collate function for sequential dataset."""
    images = []
    bboxes = []
    for images_seq, bboxes_seq in batch:
        images_seq = images_seq.transpose(0, 3, 1, 2)
        images_seq = torch.from_numpy(images_seq).div(255.0)
        bboxes_seq = torch.from_numpy(bboxes_seq)
        images.append(images_seq)
        bboxes.append(bboxes_seq)
    return images, bboxes


class RDIRDataModule(pl.LightningDataModule):
    """Sequential Data Module for training DIR model."""

    def __init__(
        self,
        data_dir: str,
        config_path: str,
        batch_size: int,
        num_workers: int = 8,
        image_size: Optional[int] = 416,
        max_boxes: int = 100,
        pin_memory: bool = True,
        prepare: bool = False,
    ):
        """
        :param data_dir: directory with dataset
        :param config_path: path to yolov4 config file
        :param batch_size: batch_size used in Data Module
        :param num_workers: number of workers used for loading data
        :param image_size: model input image size
        :param max_boxes: maximum number of boxes in a single image
        :param pin_memory: pin memory while training
        :param prepare: prepare data if necessary
        """
        super().__init__()

        self.data_dir = data_dir
        self.config_path = config_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.max_boxes = max_boxes
        self.pin_memory = pin_memory
        self.prepare = prepare

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

        self.save_hyperparameters(logger=False)

    def prepare_data(self):
        """Decompress data if necessary."""
        if not self.prepare:
            return
        data_path = Path(self.data_dir)
        files = {str(p).replace(self.data_dir, ".") for p in data_path.glob("**/*")}

        with tarfile.open(f"{self.data_dir}.tar.gz", "r:gz") as trf:
            archive_files = set(trf.getnames())
            archive_files.remove(".")

            if archive_files != files:
                logger.info("Dataset files mismatch, extracting to %s", self.data_dir)
                shutil.rmtree(data_path)
                data_path.mkdir()

                trf.extractall(self.data_dir)

    def setup(self, stage: Optional[str] = None):
        """Setup Data Module."""
        if not self.train_dataset and not self.val_dataset:
            self.train_dataset = RDIRDataset(
                dataset_dir=self.data_dir,
                config_path=self.config_path,
                files_path="train.txt",
                is_train=True,
                image_size=self.image_size,
                max_boxes=self.max_boxes,
            )
            self.val_dataset = RDIRDataset(
                dataset_dir=self.data_dir,
                config_path=self.config_path,
                files_path="val.txt",
                is_train=False,
                image_size=self.image_size,
                max_boxes=self.max_boxes,
            )

    def train_dataloader(self) -> DataLoader:
        """Prepare train DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            collate_fn=seq_collate,
        )

    def val_dataloader(self) -> DataLoader:
        """Prepare val DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=seq_collate,
        )
