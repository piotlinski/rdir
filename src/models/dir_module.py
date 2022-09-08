"""DIR model definition."""
from functools import partial
from typing import Any, Dict, List, Tuple, Type

import numpy as np
import PIL.Image as PILImage
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
import wandb
from torchmetrics import MeanSquaredError

from src.models.components.decode.decoder import DIRRepresentation
from src.models.components.decode.where import WhereTransformer
from src.models.components.encode.encoder import DIRLatents
from src.models.components.encode.seq import SeqEncoder

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
        encoder: nn.Module,
        decoder: nn.Module,
        learning_rate: float = 1e-3,
        z_present_threshold: float = -1.0,
        z_present_p_prior: float = 0.1,
        reconstruction_coef: float = 1.0,
        what_coef: float = 1.0,
        depth_coef: float = 1.0,
        present_coef: float = 1.0,
        objects_coef: float = 0.0,
        normalize_reconstructions: bool = False,
    ):
        """
        :param learning_rate: learning rate used for training the model
        :param z_present_threshold: sets z_present threshold instead of sampling
            (negative for sampling)
        :param z_present_p_prior: prior value for sampling z_present
        :param reconstruction_coef: reconstruction error component coef (entire image)
        :param what_coef: z_what distribution component coef
        :param depth_coef: z_depth distribution component coef
        :param present_coef: z_present distribution component coef
        :param objects_coef: per-object reconstruction component coef
        :param normalize_reconstructions: normalize reconstructions before scoring
        """
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.lr = learning_rate

        self.image_size = self.decoder.image_size
        self.decoded_size = self.decoder.what_dec.decoded_size
        self.drop = self.decoder.drop
        self.z_what_size = self.encoder.what_enc.latent_dim
        self.z_present_threshold = z_present_threshold
        self.z_present_p_prior = z_present_p_prior

        self._reconstruction_coef = reconstruction_coef
        self._what_coef = what_coef
        self._depth_coef = depth_coef
        self._present_coef = present_coef
        self._objects_coef = objects_coef
        self._normalize_reconstructions = normalize_reconstructions

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

    def threshold_z_present(self, z_present: torch.Tensor) -> torch.Tensor:
        """Threshold z_present to binary."""
        return torch.where(z_present > self.z_present_threshold, 1.0, 0.0)

    def encoder_forward(self, inputs: torch.Tensor) -> DIRLatents:
        """Perform forward pass through encoder network."""
        return self.encoder(inputs)

    def sample_latents(self, latents: DIRLatents) -> DIRRepresentation:
        """Sample latents to create representation."""
        (
            z_where,
            z_present,
            (z_what, z_what_scale),
            (z_depth, z_depth_scale),
        ) = latents
        self._store["z_where"] = z_where.detach()
        self._store["z_what_loc"] = z_what.detach()
        if self.is_what_probabilistic:  # else use loc
            z_what = dist.Normal(z_what, z_what_scale).sample()
        self._store["z_what"] = z_what.detach()
        self._store["z_depth_loc"] = z_depth.detach()
        if self.is_depth_probabilistic:  # else use loc
            z_depth = dist.Normal(z_depth, z_depth_scale).sample()
        self._store["z_depth"] = z_depth.detach()
        self._store["z_present_p"] = z_present.detach()
        if self.z_present_threshold < 0:
            z_present = dist.Bernoulli(z_present).sample()
        else:
            z_present = self.threshold_z_present(z_present)
        self._store["z_present"] = z_present.detach()
        return z_where, z_present, z_what, z_depth

    def decoder_forward(self, latents: DIRRepresentation) -> torch.Tensor:
        """Perform forward pass through decoder network."""
        outputs = self.decoder(
            latents,
            return_objects=True,
            normalize_reconstructions=self._normalize_reconstructions,
        )
        self._store["reconstructions"] = outputs["reconstructions"].detach()
        self._store["objects"] = outputs["objects"].detach()
        self._store["objects_where"] = outputs["objects_where"].detach()
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
        n_present = (
            torch.sum(z_present, dim=1, dtype=torch.long).squeeze(-1)
            if self.drop
            else z_present.new_tensor(
                z_present.shape[0] * [z_present.shape[1]], dtype=torch.long
            )
        )
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
        pyro.module("decoder", self.decoder)
        batch_size = x.shape[0]

        z_where, z_present, (z_what, _), (z_depth, _) = self.encoder(x)
        n_objects = z_where.shape[1]

        with pyro.plate("data", batch_size):
            # z_present
            if self.z_present_threshold < 0:
                z_present_p = x.new_full(
                    (batch_size, n_objects, 1), fill_value=self.z_present_p_prior
                )
                with poutine.scale(scale=self.present_coef(batch_size, n_objects)):
                    z_present = pyro.sample(
                        "z_present", dist.Bernoulli(z_present_p).to_event(2)
                    )
            else:
                z_present = self.threshold_z_present(z_present)

            # z_what
            if self.is_what_probabilistic:  # else use loc
                z_what_loc = x.new_zeros(batch_size, n_objects, self.z_what_size)
                z_what_scale = torch.ones_like(z_what_loc)
                with poutine.scale(scale=self.what_coef(batch_size, n_objects)):
                    z_what = pyro.sample(
                        "z_what", dist.Normal(z_what_loc, z_what_scale).to_event(2)
                    )

            # z_depth
            if self.is_depth_probabilistic:  # else use loc
                z_depth_loc = x.new_zeros(batch_size, n_objects, 1)
                z_depth_scale = torch.ones_like(z_depth_loc)
                with poutine.scale(scale=self.depth_coef(batch_size, n_objects)):
                    z_depth = pyro.sample(
                        "z_depth", dist.Normal(z_depth_loc, z_depth_scale).to_event(2)
                    )

            output = self.decoder(
                (z_where, z_present, z_what, z_depth),
                return_objects=True,
                normalize_reconstructions=self._normalize_reconstructions,
            )
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
            with pyro.plate("objects_data"):
                objects = output["objects"]
                self._store["objects"] = objects.detach()
                objects_where = output["objects_where"]
                objects_obs = self.transform_objects(x, z_present, objects_where)
                with poutine.scale(
                    scale=self.objects_coef(batch_size, objects_obs.shape[0])
                ):
                    pyro.sample(
                        "objects", dist.Bernoulli(objects).to_event(3), obs=objects_obs
                    )

    def guide(self, x: torch.Tensor):
        """Pyro guide."""
        pyro.module("encoder", self.encoder)
        batch_size = x.shape[0]

        with pyro.plate("data", batch_size):
            (
                z_where,
                z_present_p,
                (z_what, z_what_scale),
                (z_depth, z_depth_scale),
            ) = self.encoder(x)
            n_objects = z_where.shape[1]
            self._store["z_where"] = z_where.detach()
            self._store["z_present_p"] = z_present_p.detach()
            self._store["z_what_loc"] = z_what.detach()
            self._store["z_what_scale"] = z_what_scale.detach()
            self._store["z_depth_loc"] = z_depth.detach()
            self._store["z_depth_scale"] = z_depth_scale.detach()

            # z_present
            if self.z_present_threshold < 0:
                with poutine.scale(scale=self.present_coef(batch_size, n_objects)):
                    z_present = pyro.sample(
                        "z_present", dist.Bernoulli(z_present_p).to_event(2)
                    )
            else:
                z_present = self.threshold_z_present(z_present_p)
            self._store["z_present"] = z_present.detach()

            # z_what
            if self.is_what_probabilistic:  # else use loc
                with poutine.scale(scale=self.what_coef(batch_size, n_objects)):
                    z_what = pyro.sample(
                        "z_what", dist.Normal(z_what, z_what_scale).to_event(2)
                    )
            self._store["z_what"] = z_what.detach()

            # z_depth
            if self.is_depth_probabilistic:  # else use depth
                with poutine.scale(scale=self.depth_coef(batch_size, n_objects)):
                    z_depth = pyro.sample(
                        "z_depth", dist.Normal(z_depth, z_depth_scale).to_event(2)
                    )
            self._store["z_depth"] = z_depth.detach()

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
        objects = self._store["objects"]
        objects_where = self._store["objects_where"]
        objects_obs = self.transform_objects(images, z_present, objects_where)

        objects_loss = criterion(objects, objects_obs)
        self.log(f"{stage}_loss_objects", objects_loss, prog_bar=False, logger=True)

        loss = (
            self._reconstruction_coef * reconstructions_loss
            + self._objects_coef * objects_loss
        )
        self.log(f"{stage}_loss", loss, prog_bar=False, logger=True)

        return loss

    def common_run_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, stage: str
    ):
        """Common model running step for training and validation."""
        images, boxes = batch
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

        return loss

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Step for training."""
        return self.common_run_step(batch, batch_idx, stage="train")

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Step for validation."""
        loss = self.common_run_step(batch, batch_idx, stage="val")
        if batch_idx == 0 and self.logger is not None:
            to_log = {}
            with torch.no_grad():
                self.logger.experiment.log(
                    {
                        **self.visualize_inference(),
                        **self.visualize_objects(),
                        **self.log_latents(),
                    },
                    step=self.global_step,
                )
        return loss

    def visualize_inference(self) -> Dict[str, Any]:
        """Visualize model inference."""
        image = self._store["images"][[0]]
        boxes = self._store["boxes"][[0]]
        reconstruction = self._store["reconstructions"][[0]]
        z_where = self._store["z_where"][[0]]
        z_present_p = self._store["z_present_p"][[0]]
        z_depth = self._store["z_depth"][[0]]
        v_image = PILImage.fromarray(
            (image[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        )
        v_reconstruction = PILImage.fromarray(
            (reconstruction[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
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
                "box_caption": "gt_object",
            }
            for box in boxes[0]
        ]
        pred_box_data = [
            {
                "position": {
                    "middle": ((1 + where[0].item()) / 2, where[1].item()),
                    "width": where[2].item() / 2,
                    "height": where[3].item(),
                },
                "class_id": 1,
                "box_caption": "pred_object",
                "scores": {"depth": depth.item(), "present": present.item()},
            }
            for where, depth, present in zip(z_where[0], z_depth[0], z_present_p[0])
        ]
        v_boxes = {
            "gt": {"box_data": gt_box_data, "class_labels": {1: "object"}},
            "pred": {"box_data": pred_box_data, "class_labels": {1: "object"}},
        }
        return {
            "inference": wandb.Image(
                v_inference, boxes=v_boxes, caption="model inference"
            )
        }

    def visualize_objects(self) -> Dict[str, Any]:
        """Visualize reconstructed objects."""
        n_objects = 10
        z_present = self._store["z_present"][[0]]
        z_depth = self._store["z_depth"][[0]]
        n_present = z_present[0].sum(dtype=torch.long).item()
        objects = self._store["objects"][:n_present]
        objects_depth = z_depth[torch.eq(z_present, 1)][:n_present]
        _, sort_indices = torch.sort(objects_depth, descending=True)
        objects = objects.gather(
            dim=0, index=sort_indices.view(-1, 1, 1, 1).expand_as(objects)
        )
        v_objects = PILImage.new(
            "RGB",
            (objects.shape[3] * n_objects + n_objects - 1, objects.shape[2]),
            "white",
        )
        for idx, obj in enumerate(objects[:n_objects]):
            v_object = PILImage.fromarray(
                (obj.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            )
            v_objects.paste(v_object, (idx * (v_object.width + 1), 0))
        return {"objects": wandb.Image(v_objects, caption="per-object reconstructions")}

    def log_latents(self) -> Dict[str, Any]:
        """Log latents to wandb."""
        latents_names = [
            "z_where",
            "z_present_p",
            "z_what_loc",
            "z_what_scale",
            "z_depth_loc",
            "z_depth_scale",
        ]
        latents = {}
        for latent_name in latents_names:
            if (latent := self._store.get(latent_name)) is None:
                continue
            latents[latent_name] = wandb.Histogram(latent.cpu())
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


class DIRSequential(DIR):
    """Sequential DIR."""

    def __init__(
        self,
        seq_rnn_cls: Type[nn.RNNBase] = nn.GRU,
        seq_n_layers: int = 1,
        seq_bidirectional: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.seq_encoder = SeqEncoder(
            n_objects=sum(self.encoder.head.num_anchors.values()),
            z_what_size=self.z_what_size,
            what_probabilistic=self.is_what_probabilistic,
            depth_probabilistic=self.is_depth_probabilistic,
            rnn_cls=seq_rnn_cls,
            num_layers=seq_n_layers,
            bidirectional=seq_bidirectional,
        )

    @staticmethod
    def flatten_seq_dim(x: torch.Tensor) -> torch.Tensor:
        """Flatten sequence dimension."""
        return x.view(x.shape[0] * x.shape[1], *x.shape[2:])

    @staticmethod
    def add_seq_dim(x: torch.Tensor, seq_length: int) -> torch.Tensor:
        """Add sequence dimension to tensor."""
        batch_size, *shape = x.shape
        return x.view(batch_size // seq_length, seq_length, *shape)

    def encoder_forward(self, inputs: torch.Tensor) -> DIRLatents:
        """Perform forward pass through encoder network."""
        (
            z_where,
            z_present,
            (z_what, z_what_scale),
            (z_depth, z_depth_scale),
        ) = super().encoder_forward(self.flatten_seq_dim(inputs))

        unsqueeze = partial(self.add_seq_dim, seq_length=inputs.shape[1])
        latents = (
            unsqueeze(z_where),
            unsqueeze(z_present),
            (unsqueeze(z_what), unsqueeze(z_what_scale)),
            (unsqueeze(z_depth), unsqueeze(z_depth_scale)),
        )

        return self.seq_encoder(latents)

    def decoder_forward(self, latents: DIRRepresentation) -> torch.Tensor:
        """Perform forward pass through decoder network."""
        seq_length = latents[0].shape[1]
        decoded = super().decoder_forward(
            tuple(self.flatten_seq_dim(x)[0] for x in latents)
        )
        return self.add_seq_dim(decoded, seq_length)

    def model(self, x: torch.Tensor):
        """Pyro sequential model."""
        super().model(self.flatten_seq_dim(x))

    def guide(self, x: torch.Tensor):
        """Pyro sequential guide."""
        return super().guide(self.flatten_seq_dim(x))

    def common_run_step(
        self,
        batch: Tuple[List[torch.Tensor], List[torch.Tensor]],
        batch_idx: int,
        stage: str,
    ):
        """Run step including packing sequence."""
        packed = (
            rnn.pack_sequence(batch[0], enforce_sorted=False),
            rnn.pack_sequence(batch[1], enforce_sorted=False),
        )
        return super().common_run_step(packed, batch_idx, stage)
