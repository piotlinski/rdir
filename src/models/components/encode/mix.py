"""Mixer for multi-level features."""
import math
from typing import Dict

import torch
import torch.nn as nn

from src.models.components import build_conv2d_block


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
                    build_conv2d_block(
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
            downscaler = build_conv2d_block(
                larger_channels,
                downscaled_channels,
                kernel_size=3,
                padding=1,
                stride=2,
                bias=False,
            )

            upscaled_channels = larger_channels // 4
            upscaler = build_conv2d_block(
                smaller_channels,
                upscaled_channels,
                cls=nn.ConvTranspose2d,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )

            mixers[f"{larger},{smaller}"] = nn.ModuleList([downscaler, upscaler])

            channels[larger] += upscaled_channels
            channels[smaller] += downscaled_channels

        for idx, in_channels in channels.items():
            out_channels = self.out_channels[idx]
            mixers[str(idx)] = build_conv2d_block(
                in_channels, out_channels, kernel_size=1, bias=False
            )

        return nn.ModuleDict(mixers)

    def forward(self, features: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        ret = {}
        for key in self.anchors:
            ret[key] = self._downscalers[str(key)](features[key])

        iterable = list(ret.items())
        for (l_idx, l_features), (s_idx, s_features) in zip(iterable, iterable[1:]):
            downscaler, upscaler = self._mixers[f"{l_idx},{s_idx}"]
            downscaled = downscaler(l_features)
            upscaled = upscaler(s_features)

            ret[l_idx] = torch.cat([ret[l_idx], upscaled], dim=1)
            ret[s_idx] = torch.cat([ret[s_idx], downscaled], dim=1)

        for idx, feats in ret.items():
            ret[idx] = self._mixers[str(idx)](feats)

        return ret
