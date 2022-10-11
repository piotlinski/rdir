"""R-DIR model definition."""
from functools import partial
from typing import Tuple, Type

import pyro
import pyro.distributions as dist
import torch
from pyro import poutine
from torch import nn

from src.models.components.encode.encoder import RNNEncoder
from src.models.components.encode.rnn import PackedSequence, packed_forward
from src.models.components.latents import DIRLatents, DIRRepresentation
from src.models.dir_module import DIR


class RDIR(DIR):
    """Recurrent DIR."""

    def __init__(
        self,
        dir: DIR,
        n_rnn_hidden: int = 2,
        rnn_kernel_size: int = 5,
        rnn_cls: Type[nn.RNNBase] = nn.GRU,
        n_rnn_cells: int = 2,
        rnn_bidirectional: bool = False,
        train_backbone: bool = False,
        train_neck: bool = False,
        train_head: bool = False,
        train_what: bool = True,
        train_depth: bool = True,
        train_rnn: bool = True,
    ):
        rnn_encoder = RNNEncoder(
            encoder=dir.encoder,
            n_rnn_hidden=n_rnn_hidden,
            rnn_kernel_size=rnn_kernel_size,
            rnn_cls=rnn_cls,
            n_rnn_cells=n_rnn_cells,
            rnn_bidirectional=rnn_bidirectional,
            train_backbone=train_backbone,
            train_neck=train_neck,
            train_head=train_head,
            train_what=train_what,
            train_depth=train_depth,
            train_rnn=train_rnn,
        )
        super().__init__(
            encoder=rnn_encoder,
            decoder=dir.decoder,
            learning_rate=dir.lr,
            z_present_threshold=dir.z_present_threshold,
            z_present_p_prior=dir.z_present_p_prior,
            reconstruction_coef=dir._reconstruction_coef,
            what_coef=dir._what_coef,
            depth_coef=dir._depth_coef,
            present_coef=dir._present_coef,
            objects_coef=dir._objects_coef,
            normalize_reconstructions=dir._normalize_reconstructions,
            reset_non_present=dir._reset_non_present,
            negative_percentage=dir._negative_percentage,
        )

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
                objects_where = z_where.data.view(-1, z_where.shape[-1])
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

    def common_run_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        stage: str,
        store: bool = True,
    ):
        """Common model running step for training and validation."""
        images, boxes = batch

        images = nn.utils.rnn.pack_sequence(images, enforce_sorted=False)
        boxes = nn.utils.rnn.pack_sequence(boxes, enforce_sorted=False)
        self._store["images"] = images.data.detach()
        self._store["boxes"] = boxes.data.detach()

        return super().common_run_step((images, boxes), batch_idx, stage, store=False)
