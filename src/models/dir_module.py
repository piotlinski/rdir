"""DIR model definition."""
import pickle
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import PIL.Image as PILImage
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig
from torchmetrics import MeanSquaredError

from src.models.components.decode.decoder import Decoder
from src.models.components.decode.where import WhereTransformer
from src.models.components.encode.encoder import Encoder
from src.models.components.latents import (
    LatentHandler,
    DIRLatents,
    DIRRepresentation,
)

dist.enable_validation(False)


def per_site_loss(model, guide, *args, **kwargs) -> Dict[str, float]:
    """Calculate loss for each site."""
    guide_trace = poutine.trace(guide).get_trace(*args, **kwargs)
    model_trace = poutine.trace(poutine.replay(model, trace=guide_trace)).get_trace(
        *args, **kwargs
    )

    losses: Dict[str, float] = {}
    for trace in [model_trace, guide_trace]:
        for site in trace.nodes.values():
            if site["type"] == "sample" and "data" not in site["name"]:
                name = site["name"]
                elbo = losses.get(name, 0.0)
                losses[name] = elbo - site["fn"].log_prob(site["value"]).sum()

    return losses


class DIR(pl.LightningModule):
    """Detect-Infer-Repeat (per-image stage model)."""

    def __init__(
        self,
        encoder: DictConfig,
        decoder: DictConfig,
        learning_rate: float = 1e-3,
        z_present_threshold: float = 0.2,
        z_present_p_prior: float = 0.1,
        reconstruction_coef: float = 1.0,
        what_coef: float = 1.0,
        depth_coef: float = 1.0,
        present_coef: float = 1.0,
        objects_coef: float = 0.0,
        reset_non_present: bool = False,
        negative_percentage: bool = 0.1,
        max_objects: Optional[int] = 10,
    ):
        """
        :param learning_rate: learning rate used for training the model
        :param z_present_threshold: sets z_present threshold for validation
        :param z_present_p_prior: prior value for sampling z_present
        :param reconstruction_coef: reconstruction error component coef (entire image)
        :param what_coef: z_what distribution component coef
        :param depth_coef: z_depth distribution component coef
        :param present_coef: z_present distribution component coef
        :param objects_coef: per-object reconstruction component coef
        :param reset_non_present: set non-present latents to some ordinary ones
        :param negative_percentage: percentage of negative samples
        :param max_objects: max number of objects in the image (None for no limit)
        """
        super().__init__()

        self.encoder = Encoder(**encoder)
        self.decoder = Decoder(**decoder)

        self.lr = learning_rate

        self.image_size = self.decoder.image_size
        self.decoded_size = self.decoder.what_dec.decoded_size
        self.z_what_size = self.encoder.what_enc.latent_dim
        self.z_present_threshold = z_present_threshold
        self.z_present_p_prior = z_present_p_prior

        self._reconstruction_coef = reconstruction_coef
        self._what_coef = what_coef
        self._depth_coef = depth_coef
        self._present_coef = present_coef
        self._objects_coef = objects_coef

        self._reset_non_present = reset_non_present
        self._negative_percentage = negative_percentage

        self.latent_handler = LatentHandler(
            reset_non_present=reset_non_present,
            negative_percentage=negative_percentage,
            max_objects=max_objects,
        )
        self.objects_stn = WhereTransformer(image_size=self.decoded_size, inverse=True)

        self.save_hyperparameters()
        self._store: Dict[str, Any] = {}

        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.train_obj_mse = MeanSquaredError()
        self.val_obj_mse = MeanSquaredError()

        self.mse = {"train": self.train_mse, "val": self.val_mse}
        self.objects_mse = {"train": self.train_obj_mse, "val": self.val_obj_mse}

    @property
    def is_what_probabilistic(self):
        """Determines if z_what encoder is probabilistic."""
        return self.encoder.what_enc.is_probabilistic

    @property
    def is_depth_probabilistic(self):
        """Determines if z_depth encoder is probabilistic."""
        return self.encoder.depth_enc.is_probabilistic

    @property
    def is_deterministic(self):
        """Determines if DIR is deterministic."""
        return (
            (not self.is_what_probabilistic)
            and (not self.is_depth_probabilistic)
            and self.z_present_threshold > 0
        )

    @property
    def use_present_thresholding(self) -> bool:
        """Determines if z_present thresholding is used."""
        if self.encoder.nms_always:
            return True
        return not self.training

    def threshold_z_present(self, z_present: torch.Tensor) -> torch.Tensor:
        """Threshold z_present to binary."""
        return torch.where(z_present > self.z_present_threshold, 1.0, 0.0)

    def encoder_forward(self, inputs: torch.Tensor) -> DIRLatents:
        """Perform forward pass through encoder network."""
        return self.encoder(inputs)

    def _sample_where(self, z_where: torch.Tensor) -> torch.Tensor:
        return z_where

    def _sample_present(self, z_present: torch.Tensor) -> torch.Tensor:
        self._store["z_present_p"] = z_present.detach()

        if self.use_present_thresholding:
            z_present = self.threshold_z_present(z_present)
        else:
            z_present = dist.Bernoulli(z_present).sample()

        return z_present

    def _sample_what(self, z_what: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        z_what, z_what_scale = z_what

        self._store["z_what_loc"] = z_what.detach()

        if self.is_what_probabilistic:  # else use loc
            z_what = dist.Normal(z_what, z_what_scale).sample()

        return z_what

    def _sample_depth(self, z_depth: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        z_depth, z_depth_scale = z_depth

        self._store["z_depth_loc"] = z_depth.detach()

        if self.is_depth_probabilistic:  # else use loc
            z_depth = dist.Normal(z_depth, z_depth_scale).sample()

        return z_depth

    def sample_latents(self, latents: DIRLatents) -> DIRRepresentation:
        """Sample latents to create representation."""
        z_where, z_present, z_what, z_depth = self.latent_handler(
            latents,
            where_fn=lambda z_where: z_where,
            present_fn=self._sample_present,
            what_fn=self._sample_what,
            depth_fn=self._sample_depth,
        )

        self._store["z_where"] = z_where.detach()
        self._store["z_present"] = z_present.detach()
        self._store["z_what"] = z_what.detach()
        self._store["z_depth"] = z_depth.detach()

        return z_where, z_present, z_what, z_depth

    def decoder_forward(self, latents: DIRRepresentation) -> torch.Tensor:
        """Perform forward pass through decoder network."""
        outputs = self.decoder(latents)
        self._store["reconstructions"] = outputs["reconstructions"].detach()
        self._store["objects"] = outputs["objects"].detach()
        return outputs["reconstructions"]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Pass data through the autoencoder."""
        latents = self.encoder_forward(images)
        representation = self.sample_latents(latents)
        return self.decoder_forward(representation)

    def transform_objects(
        self, images: torch.Tensor, z_present: torch.Tensor, objects_where: torch.Tensor
    ) -> torch.Tensor:
        """Transform ground-truth objects with given where coordinates."""
        n_present = torch.sum(z_present.bool(), dim=1, dtype=torch.long).squeeze(-1)
        objects_indices = torch.repeat_interleave(
            torch.arange(
                n_present.numel(),
                dtype=n_present.dtype,
                device=n_present.device,
            ),
            n_present,
        )
        objects_obs = self.objects_stn(images[objects_indices], objects_where)
        self._store["objects_images"] = objects_obs.detach()
        return objects_obs

    def reconstruction_coef(self, batch_size: int) -> float:
        """Reconstruction error elbo coefficient."""
        coef = self._reconstruction_coef
        coef /= batch_size * 3 * self.image_size**2
        return coef

    def what_coef(self, batch_size: int, n_objects: int) -> float:
        """z_what sampling elbo coefficient."""
        coef = self._what_coef
        coef /= batch_size * n_objects * self.z_what_size
        return coef

    def depth_coef(self, batch_size: int, n_objects: int) -> float:
        """z_depth sampling elbo coefficient."""
        coef = self._depth_coef
        coef /= batch_size * n_objects
        return coef

    def present_coef(self, batch_size: int, n_objects: int) -> float:
        """z_present sampling elbo coefficient."""
        coef = self._present_coef
        coef /= batch_size * n_objects
        return coef

    def objects_coef(self, batch_size: int, n_objects: int) -> float:
        """Per-object reconstruction error elbo coefficient."""
        coef = self._objects_coef
        coef /= batch_size * n_objects * 3 * self.decoded_size * self.decoded_size
        return coef

    def model(self, x: torch.Tensor):
        """Pyro model."""

        def _present(z_present: torch.Tensor) -> torch.Tensor:
            n_objects = z_present.shape[1]

            if self.use_present_thresholding:
                z_present = self.threshold_z_present(z_present)
            else:
                z_present_p = x.new_full(
                    (batch_size, n_objects, 1), fill_value=self.z_present_p_prior
                )
                with poutine.scale(scale=self.present_coef(batch_size, n_objects)):
                    z_present = pyro.sample(
                        "z_present", dist.Bernoulli(z_present_p).to_event(2)
                    )

            return z_present

        def _what(z_what: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
            z_what, z_what_scale = z_what
            n_objects = z_what.shape[1]

            if self.is_what_probabilistic:  # else use loc
                z_what_loc = x.new_zeros(batch_size, n_objects, self.z_what_size)
                z_what_scale = torch.ones_like(z_what_loc)
                with poutine.scale(scale=self.what_coef(batch_size, n_objects)):
                    z_what = pyro.sample(
                        "z_what", dist.Normal(z_what_loc, z_what_scale).to_event(2)
                    )

            return z_what

        def _depth(z_depth: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
            z_depth, z_depth_scale = z_depth
            n_objects = z_depth.shape[1]

            if self.is_depth_probabilistic:  # else use loc
                z_depth_loc = x.new_zeros(batch_size, n_objects, 1)
                z_depth_scale = torch.ones_like(z_depth_loc)
                with poutine.scale(scale=self.depth_coef(batch_size, n_objects)):
                    z_depth = pyro.sample(
                        "z_depth",
                        dist.Normal(z_depth_loc, z_depth_scale).to_event(2),
                    )

            return z_depth

        pyro.module("decoder", self.decoder)
        batch_size = x.shape[0]

        latents = self.encoder(x)

        with pyro.plate("data", batch_size):
            representation = self.latent_handler(
                latents,
                where_fn=lambda z_where: z_where,
                present_fn=_present,
                what_fn=_what,
                depth_fn=_depth,
            )
            output = self.decoder(representation)
            reconstructions = output["reconstructions"]
            self._store["reconstructions"] = reconstructions.detach()

            # reconstructions
            mask = reconstructions != 0
            with poutine.scale(scale=self.reconstruction_coef(batch_size)):
                pyro.sample(
                    "reconstructions",
                    dist.Bernoulli(
                        torch.where(
                            mask, reconstructions, reconstructions.new_tensor(0.0)
                        )
                    ).to_event(3),
                    obs=torch.where(mask, x, x.new_tensor(0.0)),
                )

        # per-object reconstructions
        if self._objects_coef:
            z_where, z_present, *_ = representation
            with pyro.plate("objects_data"):
                objects = output["objects"]
                self._store["objects"] = objects.detach()
                objects_where = z_where.view(-1, z_where.shape[-1])
                objects_obs = self.transform_objects(x, z_present, objects_where)

                if not self.decoder.include_negative:
                    present_mask = (z_present > 0).view(-1, 1, 1, 1).expand_as(objects)
                    objects = objects * present_mask
                    objects_obs = objects_obs * present_mask

                with poutine.scale(
                    scale=self.objects_coef(batch_size, objects_obs.shape[0])
                ):
                    pyro.sample(
                        "objects", dist.Bernoulli(objects).to_event(3), obs=objects_obs
                    )

    def guide(self, x: torch.Tensor):
        """Pyro guide."""

        def _present(z_present: torch.Tensor) -> torch.Tensor:
            n_objects = z_present.shape[1]

            self._store["z_present_p"] = z_present.detach()

            if self.use_present_thresholding:
                z_present = self.threshold_z_present(z_present)
            else:
                with poutine.scale(scale=self.present_coef(batch_size, n_objects)):
                    z_present = pyro.sample(
                        "z_present", dist.Bernoulli(z_present).to_event(2)
                    )

            return z_present

        def _what(z_what: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
            z_what, z_what_scale = z_what

            self._store["z_what_loc"] = z_what.detach()
            self._store["z_what_scale"] = z_what_scale.detach()

            n_objects = z_what.shape[1]

            if self.is_what_probabilistic:  # else use loc
                with poutine.scale(scale=self.what_coef(batch_size, n_objects)):
                    z_what = pyro.sample(
                        "z_what", dist.Normal(z_what, z_what_scale).to_event(2)
                    )

            self._store["z_what"] = z_what.detach()

            return z_what

        def _depth(z_depth: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
            z_depth, z_depth_scale = z_depth

            self._store["z_depth_loc"] = z_depth.detach()
            self._store["z_depth_scale"] = z_depth_scale.detach()

            n_objects = z_depth.shape[1]

            if self.is_depth_probabilistic:  # else use depth
                with poutine.scale(scale=self.depth_coef(batch_size, n_objects)):
                    z_depth = pyro.sample(
                        "z_depth", dist.Normal(z_depth, z_depth_scale).to_event(2)
                    )
            self._store["z_depth"] = z_depth.detach()

            return z_depth

        pyro.module("encoder", self.encoder)
        batch_size = x.shape[0]

        with pyro.plate("data", batch_size):
            latents = self.encoder(x)
            z_where, z_present, z_what, z_depth = self.latent_handler(
                latents,
                where_fn=lambda z_where: z_where,
                present_fn=_present,
                what_fn=_what,
                depth_fn=_depth,
            )

            self._store["z_where"] = z_where.detach()
            self._store["z_present"] = z_present.detach()
            self._store["z_what"] = z_what.detach()
            self._store["z_depth"] = z_depth.detach()

            return z_where, z_present, z_what, z_depth

    def probabilistic_step(self, images: torch.Tensor, stage: str) -> torch.Tensor:
        """Training step with probabilistic model."""
        criterion = pyro.infer.Trace_ELBO().differentiable_loss
        loss = criterion(self.model, self.guide, images)
        self.log(f"{stage}_loss", loss, prog_bar=False, logger=True)

        for site, site_loss in per_site_loss(self.model, self.guide, images).items():
            self.log(f"{stage}_loss_{site}", site_loss, prog_bar=False, logger=True)

        return loss

    def deterministic_step(self, images: torch.Tensor, stage: str) -> torch.Tensor:
        """Training step with deterministic model."""
        criterion = F.mse_loss
        reconstructions = self.forward(images)

        reconstructions_loss = criterion(reconstructions, images)
        self.log(
            f"{stage}_loss_reconstructions",
            reconstructions_loss,
            prog_bar=False,
            logger=True,
        )

        z_present = self._store["z_present"]
        z_where = self._store["z_where"]
        objects = self._store["objects"]
        objects_where = z_where.view(-1, z_where.shape[-1])
        objects_obs = self.transform_objects(images, z_present, objects_where)

        if not self.decoder.include_negative:
            present_mask = (z_present > 0).view(-1, 1, 1, 1).expand_as(objects)
            objects = objects * present_mask
            objects_obs = objects_obs * present_mask

        objects_loss = criterion(objects, objects_obs)
        self.log(f"{stage}_loss_objects", objects_loss, prog_bar=False, logger=True)

        loss = (
            self._reconstruction_coef * reconstructions_loss
            + self._objects_coef * objects_loss
        )
        self.log(f"{stage}_loss", loss, prog_bar=False, logger=True)

        return loss

    def common_run_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        stage: str,
        store: bool = True,
    ):
        """Common model running step for training and validation."""
        images, boxes = batch
        if store:
            self._store["images"] = images.detach()
            self._store["boxes"] = boxes.detach()
        try:
            if self.is_deterministic:
                loss = self.deterministic_step(images, stage=stage)
            else:
                loss = self.probabilistic_step(images, stage=stage)

            self.evaluate(stage=stage)

        except ValueError:
            loss = torch.tensor(float("NaN"))

        if batch_idx == 0 and self.logger is not None:
            with torch.no_grad():
                self.logger.experiment.log(
                    {
                        **self.visualize_inference(stage),
                        **self.visualize_objects(stage),
                        **self.log_latents(stage),
                    },
                    commit=False,
                )

        return loss

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Step for training."""
        return self.common_run_step(batch, batch_idx, stage="train")

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Step for validation."""
        loss = self.common_run_step(batch, batch_idx, stage="val")
        return loss

    def visualize_inference(self, stage: str) -> Dict[str, Any]:
        """Visualize model inference."""
        n_samples = 10

        image = self._store["images"][:n_samples]
        boxes = self._store["boxes"][:n_samples]
        reconstruction = self._store["reconstructions"][:n_samples]

        positive_mask = self._store["z_present"][:n_samples] != -1
        z_where = self._store["z_where"][:n_samples]
        z_depth = self._store["z_depth"][:n_samples]

        visualizations = []
        for i in range(len(image)):
            v_image = PILImage.fromarray(
                (image[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            )
            v_reconstruction = PILImage.fromarray(
                (reconstruction[i].permute(1, 2, 0).cpu().numpy() * 255).astype(
                    np.uint8
                )
            )
            v_inference = PILImage.new(
                "RGB",
                (
                    v_image.width + v_reconstruction.width,
                    max(v_image.height, v_reconstruction.height),
                ),
                "white",
            )
            v_inference.paste(v_image, (0, 0))
            v_inference.paste(v_reconstruction, (v_image.width, 0))

            gt_box_data = [
                {
                    "position": {
                        "middle": (box[0].item() / 2, box[1].item()),
                        "width": box[2].item() / 2,
                        "height": box[3].item(),
                    },
                    "class_id": 1,
                    "box_caption": "box",
                }
                for box in boxes[i]
                if (box != 0).any()
            ]

            pred_box_data = []

            mask_i = positive_mask[i]
            z_where_i = z_where[i]
            z_depth_i = z_depth[i]
            z_where_positive = z_where_i[mask_i.expand_as(z_where_i)].view(-1, 4)
            z_where_negative = z_where_i[~mask_i.expand_as(z_where_i)].view(-1, 4)
            z_depth_positive = z_depth_i[mask_i.expand_as(z_depth_i)].view(-1, 1)
            z_depth_negative = z_depth_i[~mask_i.expand_as(z_depth_i)].view(-1, 1)
            pred_box_data.extend(
                [
                    {
                        "position": {
                            "middle": ((1 + where[0].item()) / 2, where[1].item()),
                            "width": where[2].item() / 2,
                            "height": where[3].item(),
                        },
                        "class_id": 1,
                        "box_caption": "positive",
                        "scores": {"depth": depth.item()},
                    }
                    for where, depth in zip(z_where_positive, z_depth_positive)
                ]
            )
            pred_box_data.extend(
                [
                    {
                        "position": {
                            "middle": ((1 + where[0].item()) / 2, where[1].item()),
                            "width": where[2].item() / 2,
                            "height": where[3].item(),
                        },
                        "class_id": 2,
                        "box_caption": "negative",
                        "scores": {"depth": depth.item()},
                    }
                    for where, depth in zip(z_where_negative, z_depth_negative)
                ]
            )
            v_boxes = {
                "gt": {"box_data": gt_box_data, "class_labels": {1: "object"}},
                "prediction": {
                    "box_data": pred_box_data,
                    "class_labels": {1: "positive", 2: "negative"},
                },
            }
            visualizations.append(
                wandb.Image(v_inference, boxes=v_boxes, caption="model inference")
            )
        return {f"{stage}_inference": visualizations}

    def visualize_objects(self, stage: str) -> Dict[str, Any]:
        """Visualize reconstructed objects."""
        n_samples = 10

        n_objects = 10
        z_present = self._store["z_present"][:n_samples]
        positive_mask = z_present != -1
        z_depth = self._store["z_depth"][:n_samples]
        n_present = z_present.bool().sum(dim=1, dtype=torch.long)
        objects = self._store["objects"].view(
            -1, z_present.shape[1], *self._store["objects"].shape[-3:]
        )[:n_samples]

        visualizations = []
        for i in range(len(z_present)):
            mask_i = positive_mask[i]
            z_depth_i = z_depth[i][mask_i].view(-1, 1)
            objects_i = objects[i]
            objects_mask_i = mask_i.view(*mask_i.shape, 1, 1).expand_as(objects_i)
            objects_i = objects_i[objects_mask_i].view(-1, *objects_i.shape[-3:])[
                : n_present[i]
            ]
            objects_depth_i = z_depth_i[: n_present[i]]
            _, sort_indices = torch.sort(objects_depth_i, descending=True)
            objects_i = objects_i.gather(
                dim=0, index=sort_indices.view(-1, 1, 1, 1).expand_as(objects_i)
            )
            v_objects = PILImage.new(
                "RGB",
                (objects_i.shape[3] * n_objects + n_objects - 1, objects_i.shape[2]),
                "white",
            )
            for idx, obj in enumerate(objects_i[:n_objects]):
                v_object = PILImage.fromarray(
                    (obj.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                )
                v_objects.paste(v_object, (idx * (v_object.width + 1), 0))
            visualizations.append(
                wandb.Image(v_objects, caption="reconstructed objects")
            )
        return {f"{stage}_objects": visualizations}

    def log_latents(self, stage: str) -> Dict[str, Any]:
        """Log latents to wandb."""
        latents_names = [
            f"{stage}_z_where",
            f"{stage}_z_present_p",
            f"{stage}_z_what_loc",
            f"{stage}_z_what_scale",
            f"{stage}_z_depth_loc",
            f"{stage}_z_depth_scale",
        ]
        latents = {}
        for latent_name in latents_names:
            if (latent := self._store.get(latent_name)) is None:
                continue
            latents[latent_name] = wandb.Histogram(latent.cpu())

        z_present = self._store["z_present"]
        positive = torch.count_nonzero(z_present > 0, dim=1)
        negative = torch.count_nonzero(z_present < 0, dim=1)
        total = torch.count_nonzero(z_present != 0, dim=1)
        latents[f"{stage}_n_positive"] = wandb.Histogram(positive.cpu())
        latents[f"{stage}_n_negative"] = wandb.Histogram(negative.cpu())
        latents[f"{stage}_n_total"] = wandb.Histogram(total.cpu())

        return latents

    def evaluate(self, stage: str):
        """Perform model evaluation."""
        is_train = stage == "train"
        is_val = stage == "val"
        self.mse[stage](self._store["images"], self._store["reconstructions"])
        self.log(f"{stage}_mse", self.mse[stage], on_step=is_train, on_epoch=is_val)

        self.objects_mse[stage](self._store["objects_images"], self._store["objects"])
        self.log(
            f"{stage}_objects_mse",
            self.objects_mse[stage],
            on_step=is_train,
            on_epoch=is_val,
        )

    def configure_optimizers(self):
        """Configure training optimizer."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        config = {"optimizer": optimizer}

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        config["lr_scheduler"] = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "monitor": "val_loss",
        }

        return config
