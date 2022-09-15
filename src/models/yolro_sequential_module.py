"""Sequential DIR model definition."""
from functools import partial
from typing import List, Tuple, Type, Union

import torch
import torch.nn.utils.rnn as rnn
from torch import nn

from src.models.components.decode.decoder import DIRRepresentation
from src.models.components.encode.seq import SeqEncoder, DIRLatents
from src.models.dir_module import DIR


class SequentialDIR(DIR):
    """Sequential version of DIR model."""

    def __init__(
        self,
        dir: Union[nn.Module, str],
        learning_rate: float = 1e-3,
        seq_rnn_cls: Type[nn.RNNBase] = nn.GRU,
        seq_n_layers: int = 1,
        seq_bidirectional: bool = False,
        train_encoder_backbone: bool = False,
        train_encoder_neck: bool = False,
        train_encoder_head: bool = False,
        train_encoder_what: bool = True,
        train_encoder_depth: bool = True,
        train_decoder_what: bool = True,
    ):
        """
        :param dir: DIR model to be used as a base
            (if str, it will be loaded from a checkpoint)
        :param learning_rate: learning rate used for training the model
        :param seq_rnn_cls: RNN class to use for the sequential model
        :param seq_n_layers: number of recurrent layers
        :param seq_bidirectional: use bidirectional RNN
        """
        if isinstance(dir, str):
            dir = DIR.load_from_checkpoint(dir)

        super().__init__(
            encoder=dir.encoder,
            decoder=dir.decoder,
            learning_rate=learning_rate,
            z_present_threshold=dir.z_present_threshold,
            z_present_p_prior=dir.z_present_p_prior,
            reconstruction_coef=dir.reconstruction_coef,
            what_coef=dir.what_coef,
            depth_coef=dir.depth_coef,
            present_coef=dir.present_coef,
            objects_coef=dir.objects_coef,
            normalize_reconstructions=dir._normalize_reconstructions,
        )

        self.n_objects = self._infer_n_objects()

        self.seq_encoder = SeqEncoder(
            n_objects=self.n_objects,
            z_what_size=self.z_what_size,
            what_probabilistic=self.is_what_probabilistic,
            depth_probabilistic=self.is_depth_probabilistic,
            rnn_cls=seq_rnn_cls,
            num_layers=seq_n_layers,
            bidirectional=seq_bidirectional,
        )

        self.encoder.backbone.requires_grad_(train_encoder_backbone)
        self.encoder.neck.requires_grad_(train_encoder_neck)
        self.encoder.head.requires_grad_(train_encoder_head)
        self.encoder.what_enc.requires_grad_(train_encoder_what)
        self.encoder.depth_enc.requires_grad_(train_encoder_depth)
        self.decoder.what_dec.requires_grad_(train_decoder_what)

    def _infer_n_objects(self) -> int:
        """Infer number of returned objects."""
        with torch.no_grad():
            inputs = torch.zeros(1, 3, self.image_size, self.image_size)

            features = self.encoder.backbone(inputs)
            intermediates = self.encoder.neck(features)

            _, confs = self.encoder.head(intermediates)

            return confs.shape[1]

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

    def common_run_step(
        self,
        batch: Tuple[List[torch.Tensor], List[torch.Tensor]],
        batch_idx: int,
        stage: str,
    ):
        """Run step including packing sequence."""
        packed = (
            rnn.pad_sequence(batch[0], batch_first=True).contiguous(),
            rnn.pad_sequence(batch[1], batch_first=True).contiguous(),
        )
        return super().common_run_step(packed, batch_idx, stage)
