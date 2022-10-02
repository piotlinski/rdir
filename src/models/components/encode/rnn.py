"""DIR sequential features processor."""
from typing import Dict, Tuple, Type

from torch import nn


def packed_forward(
    module: nn.Module, sequence: nn.utils.rnn.PackedSequence
) -> nn.utils.rnn.PackedSequence:
    """Forward pass of sequential data through non-sequential module.

    :param module: Module to forward pass through.
    :param sequence: Packed sequence to forward pass through module.
    :return: Packed sequence of module outputs.
    """
    data, batch_sizes, sorted_indices, unsorted_indices = sequence
    data = module(data)
    return nn.utils.rnn.PackedSequence(
        data, batch_sizes, sorted_indices, unsorted_indices
    )


class SeqRNN(nn.Module):
    """Sequential RNN for SeqEncoder."""

    def __init__(self, rnn_cls: Type[nn.RNNBase], **rnn_kwargs):
        super().__init__()

        self._rnn = rnn_cls(**rnn_kwargs)

    @staticmethod
    def preprocess_sequence(
        sequence: nn.utils.rnn.PackedSequence,
    ) -> Tuple[nn.utils.rnn.PackedSequence, Tuple[int, ...]]:
        data, batch_sizes, sorted_indices, unsorted_indices = sequence
        n_objects = data.shape[2] * data.shape[3]
        data = data.permute(0, 2, 3, 1).contiguous()
        return (
            nn.utils.rnn.PackedSequence(
                data.view(-1, data.shape[-1]),
                batch_sizes * n_objects,
                sorted_indices,
                unsorted_indices,
            ),
            data.shape,
        )

    @staticmethod
    def postprocess_sequence(
        sequence: nn.utils.rnn.PackedSequence, permuted_shape: Tuple[int, ...]
    ) -> nn.utils.rnn.PackedSequence:
        data, batch_sizes, sorted_indices, unsorted_indices = sequence
        n_objects = permuted_shape[1] * permuted_shape[2]
        data = data.view(*permuted_shape).permute(0, 3, 1, 2).contiguous()
        return nn.utils.rnn.PackedSequence(
            data, batch_sizes / n_objects, sorted_indices, unsorted_indices
        )

    def forward(self, x: nn.utils.rnn.PackedSequence) -> nn.utils.rnn.PackedSequence:
        x, permuted_shape = self.preprocess_sequence(x)
        x, _ = self._rnn(x)
        return self.postprocess_sequence(x, permuted_shape)


class SeqEncoder(nn.Module):
    """Module for processing sequential features."""

    def __init__(
        self,
        anchors: Dict[int, int],
        out_channels: Dict[int, int],
        latent_dim: int = 128,
        num_hidden: int = 2,
        kernel_size: int = 5,
        rnn_cls: Type[nn.RNNBase] = nn.GRU,
        bidirectional: bool = False,
    ):
        """"""
        super().__init__()

        self.anchors = anchors
        self.out_channels = out_channels

        self.latent_dim = latent_dim
        self.num_hidden = num_hidden
        self.kernel_size = kernel_size
        self.bidirectional = bidirectional

        self._rnn = SeqRNN(
            rnn_cls,
            input_size=latent_dim,
            hidden_size=latent_dim,
            bidirectional=bidirectional,
            batch_first=True,
            num_layers=self.num_hidden,
        )

        self._encoders = self._build_encoders()

    def _build_feature_encoder(self, in_channels: int) -> nn.ModuleList:
        """Prepare single feature sequence encoder."""
        pre = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.latent_dim,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
            ),
            nn.LeakyReLU(),
        )
        post = nn.Conv2d(
            self.latent_dim,
            self.latent_dim,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
        )
        return nn.ModuleList([pre, post])

    def _build_encoders(self) -> nn.ModuleDict:
        """Build models for recurrent encoding latent representation."""
        modules = {}
        for idx, num_anchors in self.anchors.items():
            in_channels = self.out_channels[idx]
            modules[str(idx)] = self._build_feature_encoder(in_channels)
            self.out_channels[idx] = self.latent_dim
        return nn.ModuleDict(modules)

    def forward(
        self, x: Dict[str, nn.utils.rnn.PackedSequence]
    ) -> Dict[str, nn.utils.rnn.PackedSequence]:
        ret = {}
        for idx, feature in x.items():
            pre_encoder, post_encoder = self._encoders[str(idx)]
            pre = packed_forward(pre_encoder, feature)
            seq = self._rnn(pre)
            ret[idx] = packed_forward(post_encoder, seq)
        return ret
