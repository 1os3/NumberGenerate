"""构建 MNIST 数据集与 DataLoader。

模块: data/mnist.py
依赖: torch, torchvision, config.schema, data.mnist_checks
读取配置: paths.data_dir, data.batch_size, data.num_workers, data.download, data.pin_memory, train.seed
对外接口:
    - get_mnist_loaders(cfg) -> tuple[DataLoader, DataLoader]
说明: MNIST 图像只转换为 [0,1] 张量，不做额外标准化，便于教学观察。
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
    transform = transforms.ToTensor()
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
