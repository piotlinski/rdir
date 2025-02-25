"""YOLOv4 DataModule."""
import logging
import os
import random
import shutil
import subprocess
import sys
from ast import literal_eval
from itertools import groupby
from pathlib import Path
from typing import Optional, Tuple

import albumentations as A
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

def absolute_to_relative(path: str) -> str:
    """Convert absolute path to relative path."""
    run_path = Path(sys.argv[0]).absolute().parents[1]
    return str(Path(path).relative_to(run_path))


def fetch_dvc(file: str) -> str:
    """Fetches a file using DVC and returns it."""
    logger.info(f"Fetching file {file} using DVC.")
    shutil.rmtree(file, ignore_errors=True)
    if os.path.isabs(file):  # DVC requires relative paths
        file = absolute_to_relative(file)
    subprocess.run(["dvc", "pull", file], check=True)
    return file


class XYWHDataset(Dataset):
    """YOLO-format dataset for training representation learning model."""

    def __init__(
        self,
        dataset_dir: str,
        config_path: str,
        files_path: str,
        image_size: Optional[int] = 416,
        max_boxes: int = 100,
        single_image: bool = False,
        flip: bool = False,
        crop: bool = False,
    ):
        """
        :param dataset_dir: directory in which data reside
        :param config_path: path to yolov4 model config file
        :param files_path: path to yolov4 images list
        :param image_size: model input image size
        :param max_boxes: maximum number of boxes in an image
        :param single_image: use single image from a sequence
        :param flip: flip the image randomly
        :param crop: crop the image randomly to the required size instead of resizing
        """
        super().__init__()

        self.dataset_dir = Path(dataset_dir)

        parsed = self.parse_config(config_path)
        self.width = image_size or parsed["width"]
        self.height = image_size or parsed["height"]
        self.classes = parsed["classes"]
        self.max_boxes = max_boxes

        self.resize = transforms.Resize((self.height, self.width))
        self.crop = A.Compose(
            [
                A.SmallestMaxSize(max_size=max(self.width, self.height)),
                A.RandomCrop(width=self.width, height=self.height),
            ],
            bbox_params=A.BboxParams(format="yolo"),
        )
        self.to_tensor = transforms.ToTensor()

        files = []
        with self.dataset_dir.joinpath(files_path).open("r") as fp:
            files = [str(file.replace("data/", "").strip()) for file in fp]

        if single_image:
            files = self.sample_sequences(files)

        self.files = np.array(files).astype(np.bytes_)
        self.do_flip = flip
        self.do_crop = crop

    def __len__(self):
        """Dataset length."""
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """Get single image from the dataset."""
        img_path = str(self.files[idx], encoding="utf-8")

        img, xywh = self.get_original(img_path)

        if self.do_crop:
            transformed = self.crop(image=np.array(img), bboxes=xywh)
            img = Image.fromarray(transformed["image"])
            xywh = transformed["bboxes"]
            xywh = np.array(xywh)
        else:
            img = self.resize(img)

        boxes = np.zeros([self.max_boxes, 5])
        if xywh.shape[0] > 0:
            boxes[: min(xywh.shape[0], self.max_boxes)] = xywh[
                : min(xywh.shape[0], self.max_boxes)
            ]

        if self.do_flip and random.random() > 0.5:
            img, boxes = self._horizontal_flip(img, boxes)

        return (  # need to ensure tensors are float32
            self.to_tensor(img).to(torch.float32),
            self.to_tensor(boxes).to(torch.float32)[0],
        )

    @staticmethod
    def _load_img(img_path: Path) -> Image.Image:
        """Load image from file."""
        return Image.open(img_path).convert("RGB")

    def _load_ann(self, ann_path: Path) -> np.ndarray:
        """Load annotation from file."""
        objs = []
        with ann_path.open("r") as fp:
            for line in fp:
                contents = line.split(" ")
                if contents:
                    class_id, x, y, w, h = map(literal_eval, contents)
                    x = max(1e-6, min(x, 1))
                    y = max(1e-6, min(y, 1))
                    w = max(1e-6, min(w, 1))
                    h = max(1e-6, min(h, 1))
                    if class_id < self.classes:
                        objs.append(np.array([x, y, w, h, class_id], dtype=np.float32))
        return np.vstack(objs) if objs else np.zeros((0, 5), dtype=np.float32)

    def get_original(self, img_path: str) -> Tuple[Image.Image, np.ndarray]:
        """Get original image and annotation."""
        ann_path = str(Path(img_path).with_suffix(".txt"))

        img = self._load_img(self.dataset_dir / img_path)
        xywh = self._load_ann(self.dataset_dir / ann_path)

        return img, xywh

    def _horizontal_flip(
        self, img: Image.Image, xywh: np.ndarray
    ) -> Tuple[Image.Image, np.ndarray]:
        """Flip image horizontally."""
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        xywh[:, 0] = 1 - xywh[:, 0]
        return img, xywh

    @staticmethod
    def parse_config(config_path: str):
        """Parse config file."""
        with open(config_path) as fp:
            params = {}
            to_fetch = ["[net]", "[yolo]"]
            do_fetch = False
            for line in fp:
                if do_fetch:
                    if "[" in line or line == "\n":
                        do_fetch = False
                    else:
                        key, value = map(str.strip, line.split("="))
                        try:
                            value = literal_eval(value)
                        except ValueError:
                            pass
                        params[key] = value
                if "[" in line:
                    for idx, section in enumerate(to_fetch):
                        if section in line:
                            do_fetch = True
                            to_fetch.pop(idx)
                        continue
        return params

    @staticmethod
    def sample_sequences(files: list[str]) -> list[str]:
        """Sample single image from all sequences."""
        _, lengths = zip(
            *[
                (k, len(list(v)))
                for k, v in groupby([file.rsplit("/", maxsplit=1)[0] for file in files])
            ]
        )
        min_length = min(lengths)
        center = min_length // 2
        return files[center::min_length]


class XYWHDataModule(pl.LightningDataModule):
    """Data Module for training representation learning model."""

    def __init__(
        self,
        data_dir: str,
        config_path: str,
        batch_size: int,
        num_workers: int = 8,
        size: Optional[int] = 416,
        max_boxes: int = 100,
        n_classes: Optional[int] = None,
        pin_memory: bool = True,
        prepare: bool = False,
        single_image: bool = False,
        flip: bool = False,
        crop: bool = False,
        name: str = "",
    ):
        """
        :param name: name of the dataset
        :param data_dir: directory with data
        :param config_path: path to yolov4 config file
        :param batch_size: batch_size used in Data Module
        :param num_workers: number of workers used for loading data
        :param image_size: model input image size
        :param max_boxes: maximum number of boxes in a single image
        :param n_classes: number of classes in the dataset
        :param pin_memory: pin memory while training
        :param prepare: prepare data if necessary
        :param single_image: use single image from a sequence
        :param flip: flip the image during training randomly
        :param crop: crop the image during training randomly
        """
        super().__init__()

        self.data_dir = f"{data_dir}/{name}"
        self.config_path = config_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = size
        self.max_boxes = max_boxes
        self.n_classes = n_classes
        self.pin_memory = pin_memory
        self.prepare = prepare
        self.single_image = single_image
        self.flip = flip
        self.crop = crop
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.save_hyperparameters(logger=False)

    def prepare_data(self):
        """Decompress data if necessary."""
        if not self.prepare:
            return

        logger.info("Downloading dataset from dvc")

        data_path = Path(self.data_dir)
        shutil.rmtree(data_path, ignore_errors=True)

        archive = fetch_dvc(str(data_path.with_suffix(".tar")))
        subprocess.run(["tar", "-xf", archive, "-C", data_path.parent], check=True)

    def setup(self, stage: Optional[str] = None):
        """Setup Data Module."""
        if not self.train_dataset and not self.val_dataset:
            self.train_dataset = XYWHDataset(
                dataset_dir=self.data_dir,
                config_path=self.config_path,
                files_path="train.txt",
                image_size=self.image_size,
                max_boxes=self.max_boxes,
                single_image=self.single_image,
                flip=self.flip,
                crop=self.crop,
            )
            self.val_dataset = XYWHDataset(
                dataset_dir=self.data_dir,
                config_path=self.config_path,
                files_path="val.txt",
                image_size=self.image_size,
                max_boxes=self.max_boxes,
                single_image=self.single_image,
                flip=False,
                crop=self.crop,
            )
            self.test_dataset = XYWHDataset(
                dataset_dir=self.data_dir,
                config_path=self.config_path,
                files_path="test.txt",
                image_size=self.image_size,
                max_boxes=self.max_boxes,
                single_image=self.single_image,
                flip=False,
                crop=self.crop,
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
        )

    def test_dataloader(self) -> DataLoader:
        """Prepare test DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
