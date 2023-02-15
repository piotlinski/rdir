"""DIR sequential features processor."""
from typing import Any, Dict, Tuple, Type, Union

import torch
from torch import nn

PackedSequence = nn.utils.rnn.PackedSequence
Sequence = Union[PackedSequence, Tuple[PackedSequence, ...], Dict[str, PackedSequence]]
Data = Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]


def sequence_to_tensor(sequence: Sequence) -> Tuple[Data, Tuple[Any, ...]]:
    """Convert sequence to tensor.

    :param sequence: sequence to convert
    :return: sequence data in the same format and kwargs for reconstructing sequence
    """
    args = tuple()
    if isinstance(sequence, PackedSequence):
        data, *args = sequence
        return data, args

    if isinstance(sequence, dict):
        ret = {}
        for key, value in sequence.items():
            if isinstance(value, PackedSequence):
                data, *args = value
            else:
                data, args = sequence_to_tensor(value)
            ret[key] = data
        return ret, args

    if isinstance(sequence, tuple):
        ret = []
        for element in sequence:
            if isinstance(element, PackedSequence):
                data, *args = element
            else:
                data, args = sequence_to_tensor(element)
            ret.append(data)
        return tuple(ret), args

    raise ValueError(f"Unknown sequence type: {type(sequence)}")


def tensor_to_sequence(data: Data, args: Tuple[Any, ...]) -> Sequence:
    """Convert tensor to sequence.

    :param data: processed sequence data
    :param args: args for creating sequence
    :return: processed sequence
    """
    if isinstance(data, torch.Tensor):
        return PackedSequence(data, *args)

    if isinstance(data, dict):
        ret = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                sequence = PackedSequence(value, *args)
            else:
                sequence = tensor_to_sequence(value, args)
            ret[key] = sequence
        return ret

    if isinstance(data, tuple):
        ret = []
        for element in data:
            if isinstance(element, torch.Tensor):
                sequence = PackedSequence(element, *args)
            else:
                sequence = tensor_to_sequence(element, args)
            ret.append(sequence)
        return tuple(ret)

    raise ValueError(f"Unknown data type: {type(data)}")


def packed_forward(module: nn.Module, inputs: Sequence) -> Sequence:
    """Forward pass of sequential data through non-sequential module.

    :param module: Module to forward pass through.
    :param inputs: Packed sequence to forward pass through module.
    :return: Packed sequence of module outputs.
    """
    data, args = sequence_to_tensor(inputs)
    result = module(data)
    return tensor_to_sequence(result, args)


class SeqRNN(nn.Module):
    """Sequential RNN for SeqEncoder."""

    def __init__(self, rnn_cls: Type[nn.RNNBase], **rnn_kwargs):
        super().__init__()

        self._rnn = rnn_cls(**rnn_kwargs)

    @staticmethod
    def preprocess_sequence(
        sequence: PackedSequence,
    ) -> Tuple[PackedSequence, Tuple[int, ...]]:
        data, batch_sizes, sorted_indices, unsorted_indices = sequence
        n_objects = data.shape[2] * data.shape[3]
        data = data.permute(0, 2, 3, 1).contiguous()
        return (
            PackedSequence(
                data.view(-1, data.shape[-1]),
                batch_sizes * n_objects,
                sorted_indices,
                unsorted_indices,
            ),
            data.shape,
        )

    @staticmethod
    def postprocess_sequence(
        sequence: PackedSequence, permuted_shape: Tuple[int, ...]
    ) -> PackedSequence:
        data, batch_sizes, sorted_indices, unsorted_indices = sequence
        n_objects = permuted_shape[1] * permuted_shape[2]
        data = data.view(*permuted_shape).permute(0, 3, 1, 2).contiguous()
        return PackedSequence(
            data, batch_sizes / n_objects, sorted_indices, unsorted_indices
        )

    def forward(self, x: PackedSequence) -> PackedSequence:
        x, permuted_shape = self.preprocess_sequence(x)
        x, _ = self._rnn(x)
        return self.postprocess_sequence(x, permuted_shape)


class SeqEncoder(nn.Module):
    """Module for processing sequential features."""

    def __init__(
        self,
        anchors: Dict[int, int],
        out_channels: Dict[int, int],
        num_hidden: int = 2,
        kernel_size: int = 5,
        rnn_cls: Type[nn.RNNBase] = nn.GRU,
        n_cells: int = 2,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.anchors = anchors
        self.out_channels = out_channels

        self.num_hidden = num_hidden
        self.kernel_size = kernel_size

        self.rnn_cls = rnn_cls
        self.bidirectional = bidirectional
        self.n_cells = n_cells

        self._encoders = self._build_encoders()

    def _build_feature_encoder(self, in_channels: int) -> nn.ModuleList:
        """Prepare single feature sequence encoder."""
        pre_layers = [
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(True),
        ]
        for _ in range(self.num_hidden - 1):
            pre_layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        in_channels,
                        kernel_size=self.kernel_size,
                        padding=self.kernel_size // 2,
                        bias=False,
                    ),
                    nn.BatchNorm2d(in_channels),
                    nn.LeakyReLU(True),
                ]
            )
        pre = nn.Sequential(*pre_layers)

        rnn = SeqRNN(
            self.rnn_cls,
            input_size=in_channels,
            hidden_size=in_channels // self.n_cells,
            bidirectional=self.bidirectional,
            batch_first=True,
            num_layers=self.n_cells,
        )

        post_layers = self.num_hidden * [
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(True),
        ]
        post = nn.Sequential(*post_layers)
        return nn.ModuleList([pre, rnn, post])

    def _build_encoders(self) -> nn.ModuleDict:
        """Build models for recurrent encoding latent representation."""
        modules = {}
        for idx, num_anchors in self.anchors.items():
            in_channels = self.out_channels[idx]
            modules[str(idx)] = self._build_feature_encoder(in_channels)
        return nn.ModuleDict(modules)

    def forward(self, x: Dict[str, PackedSequence]) -> Dict[str, PackedSequence]:
        ret = {}
        for idx, feature in x.items():
            pre_encoder, rnn, post_encoder = self._encoders[str(idx)]
            pre = packed_forward(pre_encoder, feature)
            seq = rnn(pre)
            ret[idx] = packed_forward(post_encoder, seq)
        return ret
