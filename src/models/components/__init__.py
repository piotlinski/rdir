from typing import Type, Union

import torch.nn as nn


def build_conv2d_block(
    in_channels: int,
    out_channels: int,
    cls: Type[Union[nn.Conv2d, nn.ConvTranspose2d]] = nn.Conv2d,
    **kwargs,
):
    """Build convolutional block."""
    return nn.Sequential(
        cls(in_channels, out_channels, **kwargs),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(True),
    )
