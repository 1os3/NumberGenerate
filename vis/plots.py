"""保存生成结果、Flow 特征 PCA 和 VAE 潜空间 PCA 可视化。

模块: vis/plots.py
依赖: pathlib, torch, matplotlib, config.schema, model, train.sampling, vis.plots_checks
读取配置: paths.output_dir, model.num_classes, sample.history_steps, sample.num_samples_per_digit, visual.pca_samples, visual.grid_columns
对外接口:
    - save_generation_steps(flow, vae, cfg) -> Path
    - save_flow_feature_pca(flow, vae, loader, cfg) -> Path
    - save_vae_latent_pca(vae, loader, cfg) -> Path
说明: 绘图库在函数内按需导入，便于缺依赖时给出明确错误。
"""

from __future__ import annotations

from pathlib import Path

import torch

from config.schema import AppConfig
from model import FlowModel, VAE
from train.sampling import sample_flow
from vis.plots_checks import check_visual_config, check_visual_images


def save_generation_steps(flow: FlowModel, vae: VAE, cfg: AppConfig) -> Path:
    """保存每个数字的一条生成轨迹网格图。"""

    check_visual_config(cfg)
    cfg.paths.output_dir.mkdir(parents=True, exist_ok=True)
    device = next(flow.parameters()).device
    labels = torch.arange(cfg.model.num_classes, device=device, dtype=torch.long)
    _, history = sample_flow(flow, vae, labels, cfg, return_history=True)
    path = cfg.paths.output_dir / "generation_steps.png"
    _save_step_grid(history, path)
    return path


@torch.no_grad()
def save_flow_feature_pca(flow: FlowModel, vae: VAE, loader, cfg: AppConfig) -> Path:
    """保存 Flow 最后一层中间特征的 PCA 散点图。"""

    check_visual_config(cfg)
    device = next(flow.parameters()).device
    images, labels = _collect_images(loader, cfg)
    images = images.to(device)
    labels = labels.to(device, dtype=torch.long)
    mu, _ = vae.encode(images)
    t = torch.full((mu.shape[0],), 0.5, device=device)
    features = flow.extract_features(mu, t, labels)[-1].mean(dim=(2, 3))
    points = _pca_2d(features.detach().cpu())
    path = cfg.paths.output_dir / "flow_feature_pca.png"
    _save_scatter(points, labels.cpu(), path, "Flow feature PCA")
    return path


@torch.no_grad()
def save_vae_latent_pca(vae: VAE, loader, cfg: AppConfig) -> Path:
    """保存 VAE 潜空间均值的 PCA 散点图。"""

    check_visual_config(cfg)
    device = next(vae.parameters()).device
    images, labels = _collect_images(loader, cfg)
    mu, _ = vae.encode(images.to(device))
    points = _pca_2d(mu.flatten(1).detach().cpu())
    path = cfg.paths.output_dir / "vae_latent_pca.png"
    _save_scatter(points, labels, path, "VAE latent PCA")
    return path


def _collect_images(loader, cfg: AppConfig) -> tuple[torch.Tensor, torch.Tensor]:
    images_list = []
    labels_list = []
    collected = 0
    for images, labels in loader:
        images_list.append(images)
        labels_list.append(labels)
        collected += images.shape[0]
        if collected >= cfg.visual.pca_samples:
            break
    images = torch.cat(images_list, dim=0)[: cfg.visual.pca_samples]
    labels = torch.cat(labels_list, dim=0)[: cfg.visual.pca_samples]
    return images, labels


def _pca_2d(features: torch.Tensor) -> torch.Tensor:
    centered = features - features.mean(dim=0, keepdim=True)
    _, _, v = torch.pca_lowrank(centered, q=2)
    return centered @ v[:, :2]


def _save_step_grid(history: torch.Tensor, path: Path) -> None:
    check_visual_images(history.reshape(-1, *history.shape[2:]))
    plt = _matplotlib()
    steps, batch = history.shape[:2]
    fig, axes = plt.subplots(steps, batch, figsize=(batch, steps))
    if steps == 1 and batch == 1:
        axes_grid = [[axes]]
    elif steps == 1:
        axes_grid = [axes]
    elif batch == 1:
        axes_grid = [[axis] for axis in axes]
    else:
        axes_grid = axes
    for row in range(steps):
        for col in range(batch):
            axis = axes_grid[row][col]
            axis.imshow(history[row, col, 0].numpy(), cmap="gray", vmin=0, vmax=1)
            axis.axis("off")
    fig.tight_layout(pad=0.1)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_scatter(points: torch.Tensor, labels: torch.Tensor, path: Path, title: str) -> None:
    plt = _matplotlib()
    fig, axis = plt.subplots(figsize=(6, 5))
    scatter = axis.scatter(points[:, 0].numpy(), points[:, 1].numpy(), c=labels.numpy(), s=8)
    axis.set_title(title)
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    fig.colorbar(scatter, ax=axis)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("缺少 matplotlib，请先安装 requirements.txt 中的依赖。") from exc
    return plt
