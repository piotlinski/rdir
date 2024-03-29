"""YOLOv4 DataModule."""
import random
import shutil
import tarfile
from ast import literal_eval
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from src.utils import get_logger
from src.vendor.yolov4.dataset import (
    fill_truth_detection,
    image_data_augmentation,
    rand_scale,
    rand_uniform_strong,
)
from src.vendor.yolov4.train import collate

logger = get_logger(__name__)


class YOLODataset(Dataset):
    """YOLO dataset for training DIR."""

    def __init__(
        self,
        dataset_dir: str,
        config_path: str,
        files_path: str,
        is_train: bool = True,
        image_size: Optional[int] = 416,
        max_boxes: int = 100,
    ):
        """
        :param dataset_dir: directory in which data reside
        :param config_path: path to yolov4 model config file
        :param files_path: path to yolov4 images list
        :param is_train: set dataset as train dataset
        :param image_size: model input image size
        :param max_boxes: maximum number of boxes in an image
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
        self.mosaic = False
        self.n_mosaic = 3
        self.max_boxes = max_boxes

        self.files = []
        with self.dataset_dir.joinpath(files_path).open("r") as fp:
            self.files = [file.replace("data/", "").strip() for file in fp]

        self._augmentation: Dict[str, Any] = {}

    def __len__(self):
        """Dataset length."""
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get single image from the dataset."""
        img_path = self.files[idx]
        if not self.is_train:
            img, xywh = self.get_original(img_path)
            img = cv2.resize(
                img, (self.width, self.height), interpolation=cv2.INTER_NEAREST
            )
        else:
            img, x1y1x2y2 = self.get_augmented(img_path, reuse=False)

            xywh = self.x1y1x2y2_to_xywh(x1y1x2y2, (self.height, self.width))

        boxes = np.zeros([self.max_boxes, 5])
        if xywh.shape[0] > 0:
            boxes[: min(xywh.shape[0], self.max_boxes)] = xywh[
                : min(xywh.shape[0], self.max_boxes)
            ]

        return img.astype(np.float32), boxes.astype(np.float32)

    @staticmethod
    def _load_img(img_path: Path) -> np.ndarray:
        """Load image from file."""
        img = cv2.imread(str(img_path))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _load_ann(self, ann_path: Path) -> np.ndarray:
        """Load annotation from file."""
        objs = []
        with ann_path.open("r") as fp:
            for line in fp:
                contents = line.split(" ")
                if contents:
                    class_id, x, y, w, h = map(literal_eval, contents)
                    if class_id < self.classes:
                        objs.append([x, y, w, h, class_id])
        return np.array(objs)

    def _get_augmentation_params(
        self, height: int, width: int, reuse: bool = False
    ) -> Dict[str, Any]:
        """Draw or reuse augmentation parameters."""
        min_offset = 0.2
        if not reuse:
            target_height, target_width = (
                np.array([height, width]) * self.jitter
            ).astype(int)
            self._augmentation = {
                "hue": rand_uniform_strong(-self.hue, self.hue),
                "saturation": rand_scale(self.saturation),
                "exposure": rand_scale(self.exposure),
                "left": random.randint(-target_width, target_width),
                "right": random.randint(-target_width, target_width),
                "top": random.randint(-target_height, target_height),
                "bottom": random.randint(-target_height, target_height),
                "cut_x": random.randint(
                    int(self.width * min_offset), int(self.width * (1 - min_offset))
                ),
                "cut_y": random.randint(
                    int(self.height * min_offset), int(self.height * (1 - min_offset))
                ),
            }
        return self._augmentation

    def _fill_truth_detection(
        self,
        x1y1x2y2: np.ndarray,
        width: int,
        height: int,
        left: int,
        right: int,
        top: int,
        bottom: int,
        **kwargs,
    ) -> np.ndarray:
        """Wrapper for fill_truth_detection."""
        x1y1x2y2, _ = fill_truth_detection(
            bboxes=x1y1x2y2,
            dx=left,
            dy=top,
            sx=width - left - right,
            sy=height - top - bottom,
            num_boxes=self.max_boxes,
            classes=self.classes,
            net_w=self.width,
            net_h=self.height,
            flip=0,
        )
        return x1y1x2y2

    def _image_data_augmentation(
        self,
        img: np.ndarray,
        x1y1x2y2: np.ndarray,
        width: int,
        height: int,
        left: int,
        right: int,
        top: int,
        bottom: int,
        hue: float,
        saturation: float,
        exposure: float,
        **kwargs,
    ) -> np.ndarray:
        """Wrapper for image_data_augmentation."""
        return image_data_augmentation(
            mat=img,
            pleft=left,
            ptop=top,
            swidth=width - left - right,
            sheight=height - top - bottom,
            dhue=hue,
            dsat=saturation,
            dexp=exposure,
            truth=x1y1x2y2,
            w=self.width,
            h=self.height,
            flip=0,
            gaussian_noise=0,
            blur=0,
        )

    def get_original(self, img_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get original image and annotation."""
        ann_path = str(Path(img_path).with_suffix(".txt"))

        img = self._load_img(self.dataset_dir / img_path)
        xywh = self._load_ann(self.dataset_dir / ann_path)

        return img, xywh

    def get_augmented(
        self, img_path: str, reuse: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get single augmented image."""
        img, xywh = self.get_original(img_path)

        height, width, _ = img.shape
        x1y1x2y2 = self.xywh_to_x1y1x2y2(xywh, (height, width))

        kwargs = {"width": width, "height": height}
        kwargs.update(self._get_augmentation_params(height, width, reuse=reuse))

        x1y1x2y2 = self._fill_truth_detection(x1y1x2y2, **kwargs)
        img = self._image_data_augmentation(img, x1y1x2y2, **kwargs)

        return img, x1y1x2y2

    @staticmethod
    def xywh_to_x1y1x2y2(
        xywh: np.ndarray, image_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """Convert XYWH boxes to X1Y1X2Y2 and (optionally) resize to image size."""
        x1y1x2y2 = xywh.copy()
        if x1y1x2y2.shape[0] == 0:
            return x1y1x2y2

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
        if xywh.shape[0] == 0:
            return xywh

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
            self.train_dataset = YOLODataset(
                dataset_dir=self.data_dir,
                config_path=self.config_path,
                files_path="train.txt",
                is_train=True,
                image_size=self.image_size,
                max_boxes=self.max_boxes,
            )
            self.val_dataset = YOLODataset(
                dataset_dir=self.data_dir,
                config_path=self.config_path,
                files_path="test.txt",
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
            collate_fn=collate,
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
            collate_fn=collate,
        )
