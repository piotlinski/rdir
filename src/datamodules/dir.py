"""DIR sequential DataModule."""
from itertools import groupby
from typing import List, Tuple

import cv2
import numpy as np

from src.datamodules.yolo import YOLODataset


class RDIRDataset(YOLODataset):
    """Sequential dataset for training DIR."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mosaic = False
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
            seq_boxes[: min(xywh.shape[0], self.max_boxes)] = xywh[
                : min(xywh.shape[0], self.max_boxes)
            ]
            images.append(img)
            boxes.append(seq_boxes)
        return np.stack(images, axis=0), np.stack(boxes, axis=0)
