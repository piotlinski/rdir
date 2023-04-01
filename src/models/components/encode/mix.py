"""Mixer for multi-level features."""
import math
from typing import Dict, Type

import torch
import torch.nn as nn


class Mixer(nn.Module):
    """Module for mixing multi-level features."""

    def __init__(self, anchors: Dict[int, int], out_channels: Dict[int, int]):
        super().__init__()

        self.anchors = anchors
        self._default_channels = out_channels
        n_channels = min([self._default_channels[key] for key in self.anchors])
        self.out_channels = {key: n_channels for key in self.anchors}

        self._downscalers = self._build_downscalers()
        self._mixers = self._build_mixers()

    @staticmethod
    def _build_conv_block(
        in_channels: int,
        out_channels: int,
        conv2d_cls: Type[nn.Module] = nn.Conv2d,
        **conv2d_kwargs,
    ):
        """Build convolutional block."""
        return nn.Sequential(
            conv2d_cls(in_channels, out_channels, **conv2d_kwargs),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(True),
        )

    def _build_downscalers(self):
        """Build downscalers for features."""
        downscalers = {}
        for key in self.anchors:
            in_channels = self._default_channels[key]
            out_channels = self.out_channels[key]
            power = int(math.log2(in_channels / out_channels))
            downscaler = []
            for _ in range(power):
                downscaler.append(
                    self._build_conv_block(
                        in_channels, in_channels // 2, kernel_size=1, bias=False
                    )
                )
                in_channels //= 2
            downscalers[str(key)] = nn.Sequential(*downscaler)
        return nn.ModuleDict(downscalers)

    def _build_mixers(self) -> nn.ModuleDict:
        """Build mixers for features."""
        mixers = {}
        iterable = list(self.anchors.keys())
        channels = {k: self.out_channels[k] for k in iterable}

        for larger, smaller in zip(iterable, iterable[1:]):
            larger_channels = self.out_channels[larger]
            smaller_channels = self.out_channels[smaller]

            downscaled_channels = smaller_channels // 4
            downscaler = self._build_conv_block(
                larger_channels,
                downscaled_channels,
                kernel_size=3,
                padding=1,
                stride=2,
                bias=False,
            )

            upscaled_channels = larger_channels // 4
            upscaler = self._build_conv_block(
                smaller_channels,
                upscaled_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
                conv2d_cls=nn.ConvTranspose2d,
            )

            mixers[f"{larger},{smaller}"] = nn.ModuleList([downscaler, upscaler])

            channels[larger] += upscaled_channels
            channels[smaller] += downscaled_channels

        for idx, in_channels in channels.items():
            out_channels = self.out_channels[idx]
            mixers[str(idx)] = self._build_conv_block(
                in_channels, out_channels, kernel_size=1, bias=False
            )

        return nn.ModuleDict(mixers)

    def forward(self, features: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        ret = {}
        for key in self.anchors:
            ret[key] = self._downscalers[str(key)](features[key])

        iterable = list(ret.items())
        for (l, l_features), (s, s_features) in zip(iterable, iterable[1:]):
            downscaler, upscaler = self._mixers[f"{l},{s}"]
            downscaled = downscaler(l_features)
            upscaled = upscaler(s_features)

            ret[l] = torch.cat([ret[l], upscaled], dim=1)
            ret[s] = torch.cat([ret[s], downscaled], dim=1)

        for idx, feats in ret.items():
            ret[idx] = self._mixers[str(idx)](feats)

        return ret
