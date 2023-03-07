import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from sklearn.manifold import TSNE
from tqdm.auto import tqdm

from src.models.rdir_module import RDIR
from src.datamodules.dir import RDIRDataModule

torch.set_grad_enabled(False)

def xywh_to_x1y1x2y2(boxes: np.ndarray) -> np.ndarray:
    """Convert XYWH boxes to corner boxes.
    :param boxes: array of XYWH boxes
    :return: array of corner boxes
    """
    center = boxes[:, :2]
    half_size = boxes[:, 2:] / 2
    return np.concatenate([center - half_size, center + half_size], axis=1)


def iou(boxes_1: np.ndarray, boxes_2: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Calculate intersection over union between two arrays of boxes.
    :param boxes_1: Mx4 array of XYWH boxes
    :param boxes_2: Nx4 array of XYWH boxes
    :param eps: epsilon for numerical stability
    :return: MxN array of IOUs between each box
    """
    boxes_1 = xywh_to_x1y1x2y2(boxes_1)
    boxes_2 = xywh_to_x1y1x2y2(boxes_2)
    tl = np.s_[..., :2]
    br = np.s_[..., 2:]
    b1 = boxes_1[:, None]
    b2 = boxes_2[None]
    intersections = (
        np.maximum(0, np.minimum(b1[br], b2[br]) - np.maximum(b1[tl], b2[tl]))
    ).prod(-1)
    unions = (b1[br] - b1[tl]).prod(-1) + (b2[br] - b2[tl]).prod(-1) - intersections
    return intersections / (unions + eps)


def load_rdir(checkpoint: str, data_dir: str, batch_size: int = 1, num_workers: int = 0):
    encoder = torch.load(checkpoint)["hyper_parameters"]["encoder"]
    encoder["yolo"] = [p.replace("/workspace", str(Path.cwd())) for p in encoder["yolo"]]
    config_path = encoder["yolo"][0]

    model = RDIR.load_from_checkpoint(checkpoint, encoder=encoder).eval()

    datamodule = RDIRDataModule(
        data_dir=data_dir,
        config_path=config_path,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        image_size=model.image_size,
    )
    return model, datamodule


def main():
    parser = argparse.ArgumentParser(
        description="Generate t-SNE embeddings for RDIR; assumes running in root directory."
    )
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("data", type=str)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


    model, datamodule = load_rdir(args.checkpoint, args.data, batch_size=args.batch_size, num_workers=args.num_workers)

    model = model.cuda()
    datamodule.setup()
    dataloader = datamodule.val_dataloader()

    representation = []
    representation_classes = []

    for images, boxes in tqdm(dataloader):
        images = nn.utils.rnn.pack_sequence(images, enforce_sorted=False)
        boxes = nn.utils.rnn.pack_sequence(boxes, enforce_sorted=False)

        latents = model.encoder_forward(images.cuda())
        (z_where, z_present, z_what, z_depth) = model.sample_latents(latents)

        boxes = boxes.data
        z_where = z_where.data
        z_what = z_what.data

        for idx in range(z_where.shape[0]):
            batch_boxes = boxes[idx]
            nonzero_boxes = batch_boxes[batch_boxes.abs().sum(dim=-1).bool()]
            xywh = nonzero_boxes[..., :4]
            classes = nonzero_boxes[..., 4].int()
            ious = iou(z_where[idx].cpu().numpy(), xywh.cpu().numpy())
            where_indices, gt_indices = linear_sum_assignment(ious, maximize=True)
            obj_what = z_what[idx][where_indices]
            obj_classes = classes[gt_indices]
            representation.append(obj_what.cpu().numpy())
            representation_classes.append(obj_classes.cpu().numpy())

    representation = np.concatenate(representation)
    representation_classes = np.concatenate(representation_classes)

    tsne = TSNE(n_components=2, n_jobs=args.num_workers, verbose=1)
    z = tsne.fit_transform(representation)
    df = pd.DataFrame()
    df["y"] = representation_classes
    df["x1"] = z[:,0]
    df["x2"] = z[:,1]

    sns.scatterplot(
        x="x1", y="x2", hue=df.y.tolist(),
        palette=sns.color_palette("hls", max(df.y) + 1),
        data=df
    ).set(title="Latent space visualization")
    plt.savefig(f"tsne_{args.checkpoint.split('/')[-1]}_{args.data.split('/')[-1]}.png")


if __name__ == "__main__":
    main()
