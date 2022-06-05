"""YOLOv4 DataModule."""
import random
import shutil
import tarfile
from ast import literal_eval
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from src.vendor.yolov4.dataset import (
    blend_truth_mosaic,
    fill_truth_detection,
    image_data_augmentation,
    rand_scale,
    rand_uniform_strong,
)
from src.vendor.yolov4.train import collate


class YOLODataset(Dataset):
    """YOLO dataset for training DIR."""

    def __init__(
        self,
        dataset_dir: str,
        config_path: str,
        files_path: str,
        is_train: bool = True,
        image_size: Optional[int] = 416,
    ):
        """
        :param dataset_dir: directory in which data reside
        :param config_path: path to yolov4 model config file
        :param files_path: path to yolov4 images list
        :param is_train: set dataset as train dataset
        :param image_size: model input image size
        """
        super().__init__()

        self.dataset_dir = Path(dataset_dir)
        self.is_train = is_train

        parsed = self.parse_config(config_path)
        self.width = image_size or parsed["width"]
        self.height = image_size or parsed["height"]
        self.angle = parsed["angle"]
        self.saturation = parsed["saturation"]
        self.exposure = parsed["exposure"]
        self.hue = parsed["hue"]
        self.classes = parsed["classes"]
        self.jitter = parsed["jitter"]
        self.mosaic = bool(parsed["mosaic"])
        self.n_mosaic = 3
        self.max_boxes = 100

        self.files = []
        with self.dataset_dir.joinpath(files_path).open("r") as fp:
            self.files = [file.replace("data/", "").strip() for file in fp]

    def __len__(self):
        """Dataset length."""
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.files[idx]
        ann_path = str(Path(img_path).with_suffix(".txt"))

        img = cv2.imread(str(self.dataset_dir / img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        xywh = self._load_ann(self.dataset_dir / ann_path)

        if self.is_train:
            mosaic = self.mosaic
            if random.randint(0, 1):
                mosaic = False

            min_offset = 0.2
            cut_x = random.randint(
                int(self.width * min_offset), int(self.width * (1 - min_offset))
            )
            cut_y = random.randint(
                int(self.height * min_offset), int(self.height * (1 - min_offset))
            )

            out_img = np.zeros((self.height, self.width, 3))
            out_x1y1x2y2 = []

            for i in range(mosaic * self.n_mosaic + 1):
                if i != 0:
                    mosaic_img_path = random.choice(self.files)
                    mosaic_ann_path = str(Path(mosaic_img_path).with_suffix(".txt"))

                    mosaic_img = cv2.imread(str(self.dataset_dir / mosaic_img_path))
                    mosaic_img = cv2.cvtColor(mosaic_img, cv2.COLOR_BGR2RGB)
                    mosaic_xywh = self._load_ann(self.dataset_dir / mosaic_ann_path)

                    img = mosaic_img
                    xywh = mosaic_xywh

                height, width, _ = img.shape
                target_height, target_width = (
                    np.array([self.height, self.width]) * self.jitter
                ).astype(int)
                x1y1x2y2 = self.xywh_to_x1y1x2y2(xywh, (height, width))

                hue = rand_uniform_strong(-self.hue, self.hue)
                saturation = rand_scale(self.saturation)
                exposure = rand_scale(self.exposure)

                left = random.randint(-target_width, target_width)
                right = random.randint(-target_width, target_width)
                top = random.randint(-target_height, target_height)
                bottom = random.randint(-target_height, target_height)

                swidth = width - left - right
                sheight = height - top - bottom

                x1y1x2y2, min_h_w = fill_truth_detection(
                    x1y1x2y2,
                    self.max_boxes,
                    self.classes,
                    0,
                    left,
                    top,
                    swidth,
                    sheight,
                    self.width,
                    self.height,
                )

                augmented = image_data_augmentation(
                    img,
                    self.width,
                    self.height,
                    left,
                    top,
                    swidth,
                    sheight,
                    0,
                    hue,
                    saturation,
                    exposure,
                    0,
                    0,
                    x1y1x2y2,
                )

                if mosaic:
                    left_shift = int(
                        min(cut_x, max(0, (-int(left) * self.width / swidth)))
                    )
                    top_shift = int(
                        min(cut_y, max(0, (-int(top) * self.height / sheight)))
                    )

                    right_shift = int(
                        min(
                            (self.width - cut_x),
                            max(0, (-int(right) * self.width / swidth)),
                        )
                    )
                    bot_shift = int(
                        min(
                            self.height - cut_y,
                            max(0, (-int(bottom) * self.height / sheight)),
                        )
                    )

                    out_img, x1y1x2y2 = blend_truth_mosaic(
                        out_img,
                        augmented,
                        x1y1x2y2.copy(),
                        self.width,
                        self.height,
                        cut_x,
                        cut_y,
                        i,
                        left_shift,
                        right_shift,
                        top_shift,
                        bot_shift,
                    )
                    out_x1y1x2y2.append(x1y1x2y2)

                else:
                    out_img = augmented
                    out_x1y1x2y2 = x1y1x2y2
            if mosaic:
                out_x1y1x2y2 = np.concatenate(out_x1y1x2y2, axis=0)

            img = out_img
            xywh = self.x1y1x2y2_to_xywh(out_x1y1x2y2, (self.height, self.width))

        else:
            img = cv2.resize(img, (self.width, self.height))

        boxes = np.zeros([self.max_boxes, 5])
        boxes[: min(xywh.shape[0], self.max_boxes)] = xywh[
            : min(xywh.shape[0], self.max_boxes)
        ]

        return img.astype(np.float32), boxes.astype(np.float32)

    @staticmethod
    def _load_ann(ann_path: Path) -> np.ndarray:
        """Load annotation from file."""
        objs = []
        with ann_path.open("r") as fp:
            for line in fp:
                contents = line.split(" ")
                if contents:
                    class_id, x, y, w, h = map(literal_eval, contents)
                    objs.append([x, y, w, h, class_id])
        return np.array(objs)

    @staticmethod
    def xywh_to_x1y1x2y2(
        xywh: np.ndarray, image_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """Convert XYWH boxes to X1Y1X2Y2 and (optionally) resize to image size."""
        x1y1x2y2 = xywh.copy()

        height, width = 1, 1
        if image_size is not None:
            height, width = image_size

        x = xywh[:, 0] * width
        y = xywh[:, 1] * height
        w = xywh[:, 2] * width
        h = xywh[:, 3] * height

        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2

        x1y1x2y2[:, :4] = np.column_stack((x1, y1, x2, y2))

        return x1y1x2y2

    @staticmethod
    def x1y1x2y2_to_xywh(
        x1y1x2y2: np.ndarray, image_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """Convert X1Y1X2Y2 boxes and (optionally) normalize to 0-1."""
        xywh = x1y1x2y2.copy()

        height, width = 1, 1
        if image_size is not None:
            height, width = image_size

        x1 = x1y1x2y2[:, 0] / width
        y1 = x1y1x2y2[:, 1] / height
        x2 = x1y1x2y2[:, 2] / width
        y2 = x1y1x2y2[:, 3] / height

        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        xywh[:, :4] = np.column_stack((x, y, w, h))

        return xywh

    @staticmethod
    def parse_config(config_path: str):
        """Parse config file."""
        with open(config_path, "r") as fp:
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


class YOLODataModule(pl.LightningDataModule):
    """Data Module for training DIR model."""

    def __init__(
        self,
        data_dir: str,
        config_path: str,
        batch_size: int,
        num_workers: int = 8,
        image_size: Optional[int] = 416,
    ):
        """
        :param data_dir: directory with dataset
        :param config_path: path to yolov4 config file
        :param batch_size: batch_size used in Data Module
        :param num_workers: number of workers used for loading data
        :param image_size: model input image size
        """
        super().__init__()

        self.data_dir = data_dir
        self.config_path = config_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

        self.save_hyperparameters(logger=False)

    def prepare_data(self):
        """Decompress data if necessary."""
        data_path = Path(self.data_dir)
        files = set(str(p).replace(self.data_dir, ".") for p in data_path.glob("**/*"))

        with tarfile.open(f"{self.data_dir}.tar.gz", "r:gz") as trf:
            archive_files = set(f for f in trf.getnames())
            archive_files.remove(".")

            if archive_files != files:
                print(archive_files ^ files)
                raise Exception
                shutil.rmtree(data_path)
                data_path.mkdir()

                trf.extractall(self.data_dir)

    def setup(self, stage: Optional[str] = None):
        """Setup Data Module."""
        if not self.train_dataset and not self.val_dataset:
            self.train_dataset = YOLODataset(
                dataset_dir=self.data_dir,
                config_path=self.config_path,
                files_path="train.txt",
                is_train=True,
                image_size=self.image_size,
            )
            self.val_dataset = YOLODataset(
                dataset_dir=self.data_dir,
                config_path=self.config_path,
                files_path="test.txt",
                is_train=False,
                image_size=self.image_size,
            )

    def train_dataloader(self) -> DataLoader:
        """Prepare train DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate,
        )

    def val_dataloader(self) -> DataLoader:
        """Prepare val DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate,
        )
