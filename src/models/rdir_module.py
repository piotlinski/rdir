"""R-DIR model definition."""
from functools import partial
import logging
from typing import Tuple

import pyro
import pyro.distributions as dist
import torch
import torch.nn.functional as F
from pyro import poutine
from torch import nn

from src.models.components.encode.encoder import RNNEncoder
from src.models.components.encode.rnn import PackedSequence, packed_forward
from src.models.components.latents import DIRLatents, DIRRepresentation
from src.models.dir_module import DIR


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RDIR(DIR):
    """Recurrent DIR."""

    def __init__(
        self,
        n_rnn_hidden: int = 2,
        rnn_kernel_size: int = 5,
        rnn_cls: str = "gru",
        n_rnn_cells: int = 2,
        rnn_bidirectional: bool = False,
        train_rnn: bool = True,
        pretrain_steps: int = 0,
        **dir_kwargs,
    ):
        super().__init__(**dir_kwargs)
        self.rnn_encoder = RNNEncoder(
            encoder=self.encoder,
            n_rnn_hidden=n_rnn_hidden,
            rnn_kernel_size=rnn_kernel_size,
            rnn_cls=rnn_cls,
            n_rnn_cells=n_rnn_cells,
            rnn_bidirectional=rnn_bidirectional,
            train_rnn=train_rnn,
        )
        self.pretrain_steps = pretrain_steps
        self.save_requires_grad()
        self._is_pretrain = False

        self.save_hyperparameters()

    def save_requires_grad(self):
        encoder = self.rnn_encoder.encoder
        self.trained = {
            "enc_backbone": next(encoder.backbone.parameters()).requires_grad,
            "enc_neck": next(encoder.neck.parameters()).requires_grad,
            "enc_head": next(encoder.head.parameters()).requires_grad,
            "enc_mixer": next(encoder.mixer.parameters()).requires_grad,
            "enc_what_enc": next(encoder.what_enc.parameters()).requires_grad,
            "enc_depth_enc": next(encoder.depth_enc.parameters()).requires_grad,
            "dec_what_dec": next(self.decoder.what_dec.parameters()).requires_grad,
        }
        if encoder.cloned_backbone is not None:
            self.trained["enc_c_backbone"] = next(
                encoder.cloned_backbone.parameters()
            ).requires_grad
        if encoder.cloned_neck is not None:
            self.trained["enc_c_neck"] = next(
                encoder.cloned_neck.parameters()
            ).requires_grad

    def set_pretrain(self):
        """Set pretrain mode."""
        logger.info("Setting pretrain mode")
        self._is_pretrain = True

        encoder = self.rnn_encoder.encoder
        encoder.backbone.requires_grad_(False)
        encoder.neck.requires_grad_(False)
        encoder.head.requires_grad_(False)
        encoder.mixer.requires_grad_(False)
        encoder.what_enc.requires_grad_(False)
        encoder.depth_enc.requires_grad_(False)
        self.decoder.what_dec.requires_grad_(False)
        if encoder.cloned_backbone is not None:
            encoder.cloned_backbone.requires_grad_(False)
        if encoder.cloned_neck is not None:
            encoder.cloned_neck.requires_grad_(False)

    def set_train(self):
        """Set train mode."""
        logger.info("Setting train mode")
        encoder = self.rnn_encoder.encoder
        encoder.backbone.requires_grad_(self.trained["enc_backbone"])
        encoder.neck.requires_grad_(self.trained["enc_neck"])
        encoder.head.requires_grad_(self.trained["enc_head"])
        encoder.mixer.requires_grad_(self.trained["enc_mixer"])
        encoder.what_enc.requires_grad_(self.trained["enc_what_enc"])
        encoder.depth_enc.requires_grad_(self.trained["enc_depth_enc"])
        self.decoder.what_dec.requires_grad_(self.trained["dec_what_dec"])
        if encoder.cloned_backbone is not None:
            encoder.cloned_backbone.requires_grad_(self.trained["enc_c_backbone"])
        if encoder.cloned_neck is not None:
            encoder.cloned_neck.requires_grad_(self.trained["enc_c_neck"])

        self._is_pretrain = False

    def encoder_forward(self, x: PackedSequence) -> DIRLatents:
        """Forward pass through the encoder."""
        return self.rnn_encoder(x)

    def sample_latents(self, latents: DIRLatents) -> DIRRepresentation:
        """Sample latents o create representation."""
        return packed_forward(super().sample_latents, latents)

    def decoder_forward(self, latents: DIRRepresentation) -> PackedSequence:
        """Forward pass through the decoder."""
        return packed_forward(super().decoder_forward, latents)

    def model(self, x: PackedSequence):
        """Pyro model."""

        def _present(z_present: torch.Tensor) -> torch.Tensor:
            n_objects = z_present.shape[1]

            if self.z_present_threshold < 0:
                z_present_p = x.data.new_full(
                    (batch_size, n_objects, 1), fill_value=self.z_present_p_prior
                )
                with poutine.scale(scale=self.present_coef(batch_size, n_objects)):
                    z_present = pyro.sample(
                        "z_present", dist.Bernoulli(z_present_p).to_event(2)
                    )
            else:
                z_present = self.threshold_z_present(z_present)

            return z_present

        def _what(z_what: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
            z_what, z_what_scale = z_what
            n_objects = z_what.shape[1]

            if self.is_what_probabilistic:  # else use loc
                z_what_loc = x.data.new_zeros(batch_size, n_objects, self.z_what_size)
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
                z_depth_loc = x.data.new_zeros(batch_size, n_objects, 1)
                z_depth_scale = torch.ones_like(z_depth_loc)
                with poutine.scale(scale=self.depth_coef(batch_size, n_objects)):
                    z_depth = pyro.sample(
                        "z_depth",
                        dist.Normal(z_depth_loc, z_depth_scale).to_event(2),
                    )

            return z_depth

        pyro.module("decoder", self.decoder)
        batch_size = x.data.shape[0]

        latents = self.encoder(x)

        with pyro.plate("data", batch_size):
            latents_handler = partial(
                self.latent_handler,
                where_fn=lambda z_where: z_where,
                present_fn=_present,
                what_fn=_what,
                depth_fn=_depth,
            )
            representation = packed_forward(latents_handler, latents)

            decoder = partial(
                self.decoder,
                return_objects=True,
                normalize_reconstructions=self._normalize_reconstructions,
            )
            output = packed_forward(decoder, representation)

            reconstructions = output["reconstructions"].data
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
                    obs=torch.where(mask, x.data, x.data.new_tensor(0.0)),
                )

        # per-object reconstructions
        if self._objects_coef:
            z_where, z_present, *_ = representation
            with pyro.plate("objects_data"):
                objects = output["objects"].data
                self._store["objects"] = objects.detach()
                objects_where = z_where.data.view(-1, z_where.data.shape[-1])
                objects_obs = self.transform_objects(
                    x.data, z_present.data, objects_where
                )
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

            if self.z_present_threshold < 0:
                with poutine.scale(scale=self.present_coef(batch_size, n_objects)):
                    z_present = pyro.sample(
                        "z_present", dist.Bernoulli(z_present).to_event(2)
                    )
            else:
                z_present = self.threshold_z_present(z_present)

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
        batch_size = x.data.shape[0]

        with pyro.plate("data", batch_size):
            latents = self.encoder(x)
            latent_handler = partial(
                self.latent_handler,
                where_fn=lambda z_where: z_where,
                present_fn=_present,
                what_fn=_what,
                depth_fn=_depth,
            )

            z_where, z_present, z_what, z_depth = packed_forward(
                latent_handler, latents
            )

            self._store["z_where"] = z_where.data.detach()
            self._store["z_present"] = z_present.data.detach()
            self._store["z_what"] = z_what.data.detach()
            self._store["z_depth"] = z_depth.data.detach()

            return z_where, z_present, z_what, z_depth

    def _store_train(self, images: torch.Tensor, boxes: torch.Tensor):
        self._store["images"] = images.data
        self._store["boxes"] = boxes.data

    def deterministic_step(self, images: torch.Tensor, stage: str) -> torch.Tensor:
        """Training step with deterministic model."""
        criterion = F.mse_loss
        reconstructions = self.forward(images)

        reconstructions_loss = criterion(reconstructions.data, images.data)
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
        objects_obs = self.transform_objects(images.data, z_present, objects_where)

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

        if self.global_step < self.pretrain_steps and not self._is_pretrain:
            self.set_pretrain()

        if self.global_step >= self.pretrain_steps and self._is_pretrain:
            self.set_train()

        images = nn.utils.rnn.pack_sequence(images)
        boxes = nn.utils.rnn.pack_sequence(boxes)

        self._store["images"] = images.data.detach()
        self._store["boxes"] = boxes.data.detach()

        return super().common_run_step((images, boxes), batch_idx, stage, store=False)
