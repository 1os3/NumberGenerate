"""提供 VAE 与 Flow 共用的卷积层和归一化层。

模块: model/layers.py
依赖: torch, torch.nn, model.layers_checks
读取配置: 无
对外接口:
    - LayerNorm2d(channels, eps=1e-6, affine=True) -> nn.Module
    - AdaLN2d(channels, condition_dim) -> nn.Module
    - ResidualBlock(channels) -> nn.Module
    - DepthwiseSeparableBlock(channels, expansion_ratio=2) -> nn.Module
    - ConditionalDepthwiseSeparableBlock(channels, condition_dim, expansion_ratio=2) -> nn.Module
说明: 普通块使用 LayerNorm2d，只有条件 Flow 主干块使用 AdaLN2d。
"""

from __future__ import annotations

import torch
from torch import nn

from model.layers_checks import (
    check_condition_tensor,
    check_expansion_ratio,
    check_layer_tensor,
    check_positive_channels,
)


class LayerNorm2d(nn.Module):
    """对 channels-first 特征图做逐像素通道 LayerNorm。"""

    def __init__(self, channels: int, eps: float = 1e-6, affine: bool = True) -> None:
        super().__init__()
        check_positive_channels(channels, "channels")
        self.channels = channels
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(channels))
            self.bias = nn.Parameter(torch.zeros(channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """归一化形状 [N,C,H,W] 的特征图。"""

        check_layer_tensor(x, self.channels, "LayerNorm2d.forward 的入参 x")
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).pow(2).mean(dim=1, keepdim=True)
        normalized = (x - mean) * torch.rsqrt(variance + self.eps)
        if not self.affine:
            return normalized
        return normalized * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class AdaLN2d(nn.Module):
    """用条件向量调制 2D LayerNorm。"""

    def __init__(self, channels: int, condition_dim: int) -> None:
        super().__init__()
        check_positive_channels(channels, "channels")
        check_positive_channels(condition_dim, "condition_dim")
        self.channels = channels
        self.condition_dim = condition_dim
        self.norm = LayerNorm2d(channels, affine=False)
        self.modulation = nn.Linear(condition_dim, channels * 2)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """根据条件向量生成缩放和平移参数。"""

        check_layer_tensor(x, self.channels, "AdaLN2d.forward 的入参 x")
        check_condition_tensor(cond, x.shape[0], self.condition_dim)
        scale, shift = self.modulation(cond).chunk(2, dim=1)
        normalized = self.norm(x)
        return normalized * (1 + scale.view(-1, self.channels, 1, 1)) + shift.view(
            -1, self.channels, 1, 1
        )


class ResidualBlock(nn.Module):
    """2D 瓶颈残差卷积块。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        check_positive_channels(channels, "channels")
        if channels < 2:
            raise ValueError("channels 必须不小于 2。")
        mid_channels = channels // 2
        self.channels = channels
        self.norm = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, mid_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.conv3 = nn.Conv2d(mid_channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行普通残差卷积。"""

        check_layer_tensor(x, self.channels, "ResidualBlock.forward 的入参 x")
        out = self.norm(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.act(out)
        out = self.conv3(out)
        return out + x


class DepthwiseSeparableBlock(nn.Module):
    """ConvNeXt 风格 2D depthwise separable 残差块。"""

    def __init__(self, channels: int, expansion_ratio: int = 2) -> None:
        super().__init__()
        check_positive_channels(channels, "channels")
        check_expansion_ratio(expansion_ratio)
        expanded_channels = channels * expansion_ratio
        self.channels = channels
        self.depthwise_conv = nn.Conv2d(
            channels, channels, kernel_size=7, padding=3, groups=channels, bias=False
        )
        self.norm = LayerNorm2d(channels)
        self.pwconv1 = nn.Conv2d(channels, expanded_channels, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(expanded_channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行不带条件的 ConvNeXt 风格残差卷积。"""

        check_layer_tensor(x, self.channels, "DepthwiseSeparableBlock.forward 的入参 x")
        out = self.depthwise_conv(x)
        out = self.norm(out)
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.pwconv2(out)
        return out + x


class ConditionalDepthwiseSeparableBlock(nn.Module):
    """在归一化处注入条件向量的 ConvNeXt 风格残差块。"""

    def __init__(self, channels: int, condition_dim: int, expansion_ratio: int = 2) -> None:
        super().__init__()
        check_positive_channels(channels, "channels")
        check_positive_channels(condition_dim, "condition_dim")
        check_expansion_ratio(expansion_ratio)
        expanded_channels = channels * expansion_ratio
        self.channels = channels
        self.condition_dim = condition_dim
        self.depthwise_conv = nn.Conv2d(
            channels, channels, kernel_size=7, padding=3, groups=channels, bias=False
        )
        self.norm = AdaLN2d(channels, condition_dim)
        self.pwconv1 = nn.Conv2d(channels, expanded_channels, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(expanded_channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """执行带 AdaLN 条件注入的 ConvNeXt 风格残差卷积。"""

        check_layer_tensor(
            x, self.channels, "ConditionalDepthwiseSeparableBlock.forward 的入参 x"
        )
        check_condition_tensor(cond, x.shape[0], self.condition_dim)
        out = self.depthwise_conv(x)
        out = self.norm(out, cond)
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.pwconv2(out)
        return out + x
