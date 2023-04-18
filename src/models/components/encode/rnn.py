"""DIR sequential features processor."""
from typing import Dict, Optional, Tuple, Type

import torch
from torch import nn

from src.models.components import build_conv2d_block


def sequence_to_tensor(sequence):
    """Convert sequence to tensor.

    :param sequence: sequence to convert
    :return: sequence data in the same format and kwargs for reconstructing sequence
    """
    args = ()
    if isinstance(sequence, nn.utils.rnn.PackedSequence):
        data, *args = sequence
        return data, args

    if isinstance(sequence, dict):
        ret = {}
        for key, value in sequence.items():
            if isinstance(value, nn.utils.rnn.PackedSequence):
                data, *args = value
            else:
                data, args = sequence_to_tensor(value)
            ret[key] = data
        return ret, args

    if isinstance(sequence, tuple):
        ret = []
        for element in sequence:
            if isinstance(element, nn.utils.rnn.PackedSequence):
                data, *args = element
            else:
                data, args = sequence_to_tensor(element)
            ret.append(data)
        return tuple(ret), args

    raise ValueError(f"Unknown sequence type: {type(sequence)}")


def tensor_to_sequence(data, args):
    """Convert tensor to sequence.

    :param data: processed sequence data
    :param args: args for creating sequence
    :return: processed sequence
    """
    if isinstance(data, torch.Tensor):
        return nn.utils.rnn.PackedSequence(data, *args)

    if isinstance(data, dict):
        ret = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                sequence = nn.utils.rnn.PackedSequence(value, *args)
            else:
                sequence = tensor_to_sequence(value, args)
            ret[key] = sequence
        return ret

    if isinstance(data, tuple):
        ret = []
        for element in data:
            if isinstance(element, torch.Tensor):
                sequence = nn.utils.rnn.PackedSequence(element, *args)
            else:
                sequence = tensor_to_sequence(element, args)
            ret.append(sequence)
        return tuple(ret)

    raise ValueError(f"Unknown data type: {type(data)}")


def packed_forward(module: nn.Module, inputs):
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

        self._post: Optional[nn.Module] = None
        if rnn_kwargs["bidirectional"]:
            self._post = nn.Linear(
                rnn_kwargs["hidden_size"] * 2, rnn_kwargs["hidden_size"]
            )

    @staticmethod
    def preprocess_sequence(
        sequences: Dict[str, nn.utils.rnn.PackedSequence],
    ) -> Tuple[nn.utils.rnn.PackedSequence, Dict[str, torch.Size]]:
        shapes = {}
        datas = []
        lengths = []

        for key, seq in sequences.items():
            data, seq_lengths = nn.utils.rnn.pad_packed_sequence(seq, batch_first=True)
            data = data.permute(0, 3, 4, 1, 2).contiguous()
            shapes[key] = data.shape
            datas.append(data.view(-1, *data.shape[-2:]))
            lengths.append(seq_lengths.repeat(data.shape[1] * data.shape[2]))

        return (
            nn.utils.rnn.pack_padded_sequence(
                torch.cat(datas, dim=0), lengths=torch.cat(lengths), batch_first=True
            ),
            shapes,
        )

    @staticmethod
    def postprocess_sequence(
        sequence: nn.utils.rnn.PackedSequence,
        permuted_shapes: Dict[str, Tuple[int, ...]],
    ) -> Dict[str, nn.utils.rnn.PackedSequence]:
        ret = {}

        data, seq_lenghts = nn.utils.rnn.pad_packed_sequence(sequence, batch_first=True)

        splits = [s[0] * s[1] * s[2] for s in permuted_shapes.values()]
        s_data = torch.split(data, splits, dim=0)
        s_lengths = torch.split(seq_lenghts, splits, dim=0)

        for (k, shape), k_data, k_lengths in zip(
            permuted_shapes.items(), s_data, s_lengths
        ):
            k_data = k_data.view(*shape).permute(0, 3, 4, 1, 2).contiguous()
            k_lengths = k_lengths[: shape[0]]
            ret[k] = nn.utils.rnn.pack_padded_sequence(
                k_data, lengths=k_lengths, batch_first=True
            )
        return ret

    def forward(
        self, x: Dict[str, nn.utils.rnn.PackedSequence]
    ) -> Dict[str, nn.utils.rnn.PackedSequence]:
        x, permuted_shape = self.preprocess_sequence(x)
        x, _ = self._rnn(x)
        if self._post is not None:
            x = packed_forward(self._post, x)  # type: ignore
        return self.postprocess_sequence(x, permuted_shape)


class SeqEncoder(nn.Module):
    """Module for processing sequential features."""

    RNNS = {"gru": nn.GRU, "lstm": nn.LSTM}

    def __init__(
        self,
        anchors: Dict[int, int],
        out_channels: Dict[int, int],
        num_hidden: int = 2,
        kernel_size: int = 5,
        rnn_cls: str = "gru",
        n_cells: int = 2,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.anchors = anchors
        self.out_channels = out_channels

        self.num_hidden = num_hidden
        self.kernel_size = kernel_size

        self.rnn_cls = self.RNNS[rnn_cls]
        self.bidirectional = bidirectional
        self.n_cells = n_cells

        self._encoders = self._build_encoders()

    def _build_hidden(self, channels: int) -> nn.Module:
        """Prepare single hidden conv layers set."""
        modules = []
        for _ in range(self.num_hidden):
            modules.append(
                build_conv2d_block(
                    channels,
                    channels,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                    bias=False,
                )
            )
        return nn.Sequential(*modules)

    def _build_feature_encoders(self, channels: int) -> nn.ModuleList:
        """Prepare single feature encoders."""
        return nn.ModuleList(
            [
                self._build_hidden(channels),
                self._build_hidden(channels),
            ]
        )

    def _build_encoders(self) -> nn.ModuleDict:
        """Build models for recurrent encoding latent representation."""
        modules = {}
        for idx in self.anchors:
            channels = self.out_channels[idx]
            pre, post = self._build_feature_encoders(channels)
            modules[f"{idx}_pre"] = pre
            modules[f"{idx}_post"] = post

        channels = set(self.out_channels.values()).pop()
        modules["rnn"] = SeqRNN(
            self.rnn_cls,
            input_size=channels,
            hidden_size=channels,
            bidirectional=self.bidirectional,
            batch_first=True,
            num_layers=self.n_cells,
        )

        return nn.ModuleDict(modules)

    def forward(
        self, x: Dict[str, nn.utils.rnn.PackedSequence]
    ) -> Dict[str, nn.utils.rnn.PackedSequence]:
        ret = {}

        for idx, feature in x.items():
            ret[idx] = packed_forward(self._encoders[f"{idx}_pre"], feature)

        ret = self._encoders["rnn"](ret)

        for idx, feature in ret.items():
            ret[idx] = packed_forward(self._encoders[f"{idx}_post"], feature)

        return ret
