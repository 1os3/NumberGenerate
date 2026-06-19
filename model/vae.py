"""定义无条件卷积 VAE。

模块: model/vae.py
依赖: torch, torch.nn, config.schema, model.layers, model.vae_checks
读取配置: model.image_channels, model.image_size, model.vae_hidden_channels, model.latent_channels, model.latent_size
对外接口:
    - VAE(cfg) -> nn.Module
    - VAE.encode(x) -> tuple[Tensor, Tensor]
    - VAE.decode(z) -> Tensor
    - VAE.forward(x) -> tuple[Tensor, Tensor, Tensor]
说明: VAE 不接收时间或数字标签，全部使用普通 LayerNorm2d 和卷积。
"""

from __future__ import annotations

import torch
from torch import nn

from config.schema import AppConfig
from model.layers import ResidualBlock
from model.vae_checks import check_image_batch, check_latent_batch, check_vae_config


class VAEEncoder(nn.Module):
    """将 MNIST 图像编码为潜空间高斯分布参数。"""

    def __init__(self, cfg: AppConfig) -> None:
        super().__init__()
        hidden = cfg.model.vae_hidden_channels
        latent = cfg.model.latent_channels
        self.cfg = cfg
        self.stem = nn.Conv2d(cfg.model.image_channels, hidden, kernel_size=1)
        self.block = ResidualBlock(hidden)
        self.downsample = nn.Conv2d(hidden, latent, kernel_size=2, stride=2)
        self.to_mu = nn.Conv2d(latent, latent, kernel_size=1)
        self.to_logvar = nn.Conv2d(latent, latent, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """输出潜变量分布的均值与对数方差。"""

        check_image_batch(x, self.cfg)
        hidden = self.stem(x)
        hidden = self.block(hidden)
        latent = self.downsample(hidden)
        return self.to_mu(latent), self.to_logvar(latent)


class VAEDecoder(nn.Module):
    """将潜变量解码回 MNIST 图像空间。"""

    def __init__(self, cfg: AppConfig) -> None:
        super().__init__()
        hidden = cfg.model.vae_hidden_channels
        latent = cfg.model.latent_channels
        self.cfg = cfg
        self.block = ResidualBlock(latent)
        self.expand = nn.Conv2d(latent, hidden * 4, kernel_size=1)
        self.upsample = nn.PixelShuffle(2)
        self.out = nn.Conv2d(hidden, cfg.model.image_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """输出范围在 [0,1] 的重建图像。"""

        check_latent_batch(z, self.cfg)
        hidden = self.block(z)
        hidden = self.expand(hidden)
        hidden = self.upsample(hidden)
        return torch.sigmoid(self.out(hidden))


class VAE(nn.Module):
    """组合 Encoder、重参数化采样和 Decoder 的无条件 VAE。"""

    def __init__(self, cfg: AppConfig) -> None:
        super().__init__()
        check_vae_config(cfg)
        self.cfg = cfg
        self.encoder = VAEEncoder(cfg)
        self.decoder = VAEDecoder(cfg)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """根据输入图像输出潜变量均值与对数方差。"""

        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """根据潜变量重建图像。"""

        return self.decoder(z)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """用重参数化技巧从潜变量分布采样。"""

        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """执行 VAE 前向传播并返回重建、均值和对数方差。"""

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
