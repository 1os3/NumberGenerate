"""保存生成结果、Flow 特征图、Flow 特征 PCA 和 VAE 潜空间 PCA 可视化。

模块: vis/plots.py
依赖: pathlib, torch, matplotlib, config.schema, model, train.flow_trainer, train.sampling, vis.plots_checks
读取配置: paths.output_dir, model.num_classes, sample.history_steps, sample.sampling_steps, visual.pca_samples, visual.feature_map_channels, visual.feature_map_time
对外接口:
    - save_generation_steps(flow, vae, cfg) -> Path
    - save_flow_feature_maps(flow, vae, loader, cfg) -> Path
    - save_flow_feature_pca(flow, vae, loader, cfg) -> Path
    - save_vae_latent_pca(vae, loader, cfg) -> Path
说明: 绘图库在函数内按需导入，便于缺依赖时给出明确错误。
"""

from __future__ import annotations

from pathlib import Path

import torch

from config.schema import AppConfig
from model import FlowModel, VAE
from train.flow_trainer import sample_vae_posterior
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
def save_flow_feature_maps(flow: FlowModel, vae: VAE, loader, cfg: AppConfig) -> Path:
    """保存单个样本在 Flow 各主干块后的真实中间层特征图。"""

    check_visual_config(cfg)
    cfg.paths.output_dir.mkdir(parents=True, exist_ok=True)
    device = next(flow.parameters()).device
    images, labels = _first_batch(loader)
    image = images[:1].to(device)
    label = labels[:1].to(device, dtype=torch.long)
    z = sample_vae_posterior(vae, image)
    t = torch.full((1,), cfg.visual.feature_map_time, device=device)
    features = flow.extract_features(z, t, label)
    feature_maps = [
        feature[0, : cfg.visual.feature_map_channels].detach().cpu() for feature in features
    ]
    path = cfg.paths.output_dir / "flow_feature_maps.png"
    _save_feature_map_grid(feature_maps, label.item(), cfg.visual.feature_map_time, path)
    return path


@torch.no_grad()
def save_flow_feature_pca(flow: FlowModel, vae: VAE, loader, cfg: AppConfig) -> Path:
    """保存 Flow 最后一层中间特征的 PCA 散点图。"""

    check_visual_config(cfg)
    cfg.paths.output_dir.mkdir(parents=True, exist_ok=True)
    device = next(flow.parameters()).device
    images, labels = _collect_images(loader, cfg)
    images = images.to(device)
    labels = labels.to(device, dtype=torch.long)
    z = sample_vae_posterior(vae, images)
    t = torch.full((z.shape[0],), 0.5, device=device)
    features = flow.extract_features(z, t, labels)[-1].mean(dim=(2, 3))
    points = _pca_2d(features.detach().cpu())
    path = cfg.paths.output_dir / "flow_feature_pca.png"
    _save_scatter(points, labels.cpu(), path, "Flow feature PCA")
    return path


@torch.no_grad()
def save_vae_latent_pca(vae: VAE, loader, cfg: AppConfig) -> Path:
    """保存 VAE 潜空间均值的 PCA 散点图。"""

    check_visual_config(cfg)
    cfg.paths.output_dir.mkdir(parents=True, exist_ok=True)
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


def _first_batch(loader) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        return next(iter(loader))
    except StopIteration as exc:
        raise ValueError("可视化需要至少一个数据 batch。") from exc


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


def _save_feature_map_grid(
    feature_maps: list[torch.Tensor], label: int, time_value: float, path: Path
) -> None:
    plt = _matplotlib()
    rows = len(feature_maps)
    cols = feature_maps[0].shape[0]
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
    axes_grid = _axes_grid(axes, rows, cols)
    for row, maps in enumerate(feature_maps):
        for col, feature_map in enumerate(maps):
            axis = axes_grid[row][col]
            axis.imshow(_normalize_map(feature_map).numpy(), cmap="viridis")
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(f"C{col}", fontsize=7)
            if col == 0:
                axis.set_ylabel(f"B{row + 1}", fontsize=8)
    fig.suptitle(f"Flow feature maps | label={label} | t={time_value:.2f}", fontsize=10)
    fig.tight_layout(pad=0.1)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _axes_grid(axes, rows: int, cols: int):
    if rows == 1 and cols == 1:
        return [[axes]]
    if rows == 1:
        return [axes]
    if cols == 1:
        return [[axis] for axis in axes]
    return axes


def _normalize_map(feature_map: torch.Tensor) -> torch.Tensor:
    min_value = feature_map.min()
    max_value = feature_map.max()
    return (feature_map - min_value) / (max_value - min_value + 1e-6)


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
