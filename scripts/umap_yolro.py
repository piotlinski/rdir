import argparse
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm

from src.datamodules.yolo import YOLODataModule
from src.models.dir_module import DIR
from umap import UMAP  # type: ignore

torch.set_grad_enabled(False)


def xywh_to_x1y1x2y2(boxes):
    """Convert XYWH boxes to corner boxes.

    :param boxes: array of XYWH boxes
    :return: array of corner boxes
    """
    center = boxes[:, :2]
    half_size = boxes[:, 2:] / 2
    return np.concatenate([center - half_size, center + half_size], axis=1)


def iou(boxes_1, boxes_2, eps=1e-5):
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


def load_dir(checkpoint, data_dir, batch_size=1, num_workers=0, fallback_config=""):
    try:
        encoder = torch.load(checkpoint)["hyper_parameters"]["encoder"]
        encoder["yolo"] = [
            p.replace("/workspace", str(Path.cwd())) for p in encoder["yolo"]
        ]
        config_path = encoder["yolo"][0]
    except (KeyError, TypeError):
        config_path = fallback_config

    model = DIR.load_from_checkpoint(checkpoint, encoder=encoder).eval()
    datamodule = YOLODataModule(
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
        description=(
            "Generate t-SNE embeddings for DIR; assumes running in root directory."
        )
    )
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--fallback_config", type=str, default="")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("data", type=str)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    model, datamodule = load_dir(
        args.checkpoint,
        args.data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        fallback_config=args.fallback_config,
    )

    model = model.cuda()
    datamodule.setup()
    dataloader = datamodule.val_dataloader()

    representation = []
    representation_classes = []

    for images, boxes in tqdm(dataloader):
        latents = model.encoder_forward(images.cuda())
        (z_where, z_present, z_what, z_depth) = model.sample_latents(latents)
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

    reducer = UMAP(n_components=2, n_jobs=args.num_workers, verbose=1)
    z = reducer.fit_transform(representation)
    df = pd.DataFrame()
    df["y"] = representation_classes
    df["x1"] = z[:, 0]
    df["x2"] = z[:, 1]

    sns.scatterplot(
        x="x1",
        y="x2",
        hue=df.y.tolist(),
        palette=sns.color_palette("hls", max(df.y) + 1),
        data=df,
        size=1,
    ).set(title="Latent space visualization")

    filename = f"umap_{args.checkpoint.split('/')[-1]}_{args.data.split('/')[-1]}"
    plt.savefig(f"{filename}.png")
    with open(f"{filename}.pickle", "wb") as fp:
        pickle.dump([representation, representation_classes], fp)
    with open(f"{filename}_reducer.pickle", "wb") as fp:
        pickle.dump(reducer, fp)


if __name__ == "__main__":
    main()
