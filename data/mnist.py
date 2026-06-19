"""构建 MNIST 数据集与 DataLoader。

模块: data/mnist.py
依赖: torch, torchvision, config.schema, data.mnist_checks
读取配置: paths.data_dir, data.batch_size, data.num_workers, data.download, data.pin_memory, data.binarize_threshold, train.seed
对外接口:
    - get_mnist_loaders(cfg) -> tuple[DataLoader, DataLoader]
    - mnist_to_presence_channels(image, threshold) -> Tensor
说明: MNIST 图像被转换为 [background, foreground] 二通道 one-hot 张量，便于 VAE 学习明确的有/无语义。
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config.schema import AppConfig
from data.mnist_checks import check_mnist_config


def get_mnist_loaders(cfg: AppConfig) -> tuple[DataLoader, DataLoader]:
    """创建训练集和测试集 DataLoader。

    参数:
        cfg: 项目配置对象，读取 MNIST 路径与加载参数
    返回:
        训练 DataLoader 与测试 DataLoader
    """

    check_mnist_config(cfg)
    cfg.paths.data_dir.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(
                lambda image: mnist_to_presence_channels(
                    image, cfg.data.binarize_threshold
                )
            ),
        ]
    )
    train_set = datasets.MNIST(
        root=str(cfg.paths.data_dir),
        train=True,
        download=cfg.data.download,
        transform=transform,
    )
    test_set = datasets.MNIST(
        root=str(cfg.paths.data_dir),
        train=False,
        download=cfg.data.download,
        transform=transform,
    )
    generator = torch.Generator().manual_seed(cfg.train.seed)
    loader_kwargs = {
        "batch_size": cfg.data.batch_size,
        "num_workers": cfg.data.num_workers,
        "pin_memory": cfg.data.pin_memory,
        "persistent_workers": cfg.data.num_workers > 0,
    }
    return (
        DataLoader(train_set, shuffle=True, generator=generator, **loader_kwargs),
        DataLoader(test_set, shuffle=False, **loader_kwargs),
    )


def mnist_to_presence_channels(image: torch.Tensor, threshold: float) -> torch.Tensor:
    """将单通道 MNIST 图像转换为 background/foreground 二通道表示。

    参数:
        image: 形状 [1,H,W]、范围 [0,1] 的 MNIST 图像
        threshold: 二值化阈值，大于该值表示 foreground
    返回:
        形状 [2,H,W] 的 one-hot 张量，第 0 通道是 background，第 1 通道是 foreground
    """

    foreground = (image > threshold).to(dtype=image.dtype)
    background = 1.0 - foreground
    return torch.cat([background, foreground], dim=0)
