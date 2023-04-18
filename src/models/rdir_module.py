"""R-DIR model definition."""
import logging
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from src.models.components.encode.encoder import RNNEncoder
from src.models.components.encode.rnn import packed_forward
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

    def encoder_forward(self, x: nn.utils.rnn.PackedSequence) -> DIRLatents:
        """Forward pass through the encoder."""
        return self.rnn_encoder(x)

    def sample_latents(self, latents: DIRLatents) -> DIRRepresentation:
        """Sample latents o create representation."""
        return packed_forward(super().sample_latents, latents)

    def decoder_forward(
        self, latents: DIRRepresentation
    ) -> nn.utils.rnn.PackedSequence:
        """Forward pass through the decoder."""
        return packed_forward(super().decoder_forward, latents)

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
