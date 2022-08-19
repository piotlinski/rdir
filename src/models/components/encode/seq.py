"""DIR sequential layer."""
from typing import Optional, Tuple, Type

import torch
import torch.nn as nn

DIRLatents = Tuple[
    torch.Tensor,
    torch.Tensor,
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
]


class SeqEncoder(nn.Module):
    """Recurrent layer for learning representation of objects in sequences."""

    def __init__(
        self,
        n_objects: int,
        z_what_size: int,
        what_probabilistic: bool,
        depth_probabilistic: bool,
        rnn_cls: Type[nn.RNNBase] = nn.GRU,
        num_layers: int = 1,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.n_objects = n_objects
        self.rnn_cls = rnn_cls
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.what_rnn = self._build_rnn(z_what_size)
        self.what_scale_rnn: Optional[nn.RNNBase] = None
        if what_probabilistic:
            self.what_scale_rnn = self._build_rnn(z_what_size)

        self.depth_rnn = self._build_rnn(1)
        self.depth_scale_rnn: Optional[nn.RNNBase] = None
        if depth_probabilistic:
            self.depth_scale_rnn = self._build_rnn(1)

    def _build_rnn(self, latent_dim: int) -> nn.RNNBase:
        """Build recurrent network for learning latents."""
        return self.rnn_cls(
            self.n_objects * latent_dim,
            self.n_objects * latent_dim,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            batch_first=True,
        )

    @staticmethod
    def _flat_forward(x: torch.Tensor, rnn: nn.RNNBase) -> torch.Tensor:
        """Flatten input and forward through recurrent network."""
        original = x.shape
        x = x.view(x.shape[0] * x.shape[1], -1)
        x, _ = rnn(x)
        x = x.view(*original)
        return x

    def forward(self, latents: DIRLatents) -> DIRLatents:
        """Pass latents through RNN."""
        (
            z_where,
            z_present,
            (z_what, z_what_scale),
            (z_depth, z_depth_scale),
        ) = latents

        z_what = self._flat_forward(z_what, self.what_rnn)
        if self.what_scale_rnn is not None:
            z_what_scale = self._flat_forward(z_what_scale, self.what_scale_rnn)
        z_depth = self._flat_forward(z_depth, self.depth_rnn)
        if self.depth_scale_rnn is not None:
            z_depth_scale = self._flat_forward(z_depth_scale, self.depth_scale_rnn)

        return (
            z_where,
            z_present,
            (z_what, z_what_scale),
            (z_depth, z_depth_scale),
        )
