"""保存生成结果、Flow 特征 PCA 图、VAE 重构、VAE PCA 和潜变量分布可视化。

模块: vis/plots.py
依赖: pathlib, torch, matplotlib, config.schema, model, train.flow_trainer, train.sampling, vis.plots_checks
读取配置: paths.output_dir, model.num_classes, sample.history_steps, sample.sampling_steps, visual.pca_samples, visual.feature_map_channels, visual.feature_map_time
对外接口:
    - save_generation_steps(flow, vae, cfg) -> Path
    - save_flow_feature_pca_map(flow, vae, loader, cfg) -> Path
    - save_flow_feature_pca(flow, vae, loader, cfg) -> Path
    - save_vae_reconstruction(vae, loader, cfg) -> Path
    - save_vae_latent_pca(vae, loader, cfg) -> Path
    - save_vae_latent_distribution(vae, loader, cfg) -> Path
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
def save_flow_feature_pca_map(flow: FlowModel, vae: VAE, loader, cfg: AppConfig) -> Path:
    """保存单个样本在 Flow 末端特征上的三通道 PCA 图。"""

    check_visual_config(cfg)
    cfg.paths.output_dir.mkdir(parents=True, exist_ok=True)
    device = next(flow.parameters()).device
    images, labels = _first_batch(loader)
    image = images[:1].to(device)
    label = labels[:1].to(device, dtype=torch.long)
    z = sample_vae_posterior(vae, image)
    t = torch.full((1,), cfg.visual.feature_map_time, device=device)
    features = flow.extract_features(z, t, label)
    rgb_map = _feature_map_pca_rgb(
        features[-1][0].detach().cpu(), cfg.visual.feature_map_channels
    )
    path = cfg.paths.output_dir / "flow_feature_pca_map.png"
    _save_rgb_map(rgb_map, path, f"Flow final feature PCA | label={label.item()} | t={cfg.visual.feature_map_time:.2f}")
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


@torch.no_grad()
def save_vae_reconstruction(vae: VAE, loader, cfg: AppConfig) -> Path:
    """保存单张 MNIST 图像的 VAE 压缩、重构和误差图。"""

    check_visual_config(cfg)
    cfg.paths.output_dir.mkdir(parents=True, exist_ok=True)
    device = next(vae.parameters()).device
    images, labels = _first_batch(loader)
    image = images[:1].to(device)
    label = int(labels[0].item())
    mu, _ = vae.encode(image)
    reconstruction = vae.decode(mu).detach().cpu()
    latent_rgb = _feature_map_pca_rgb(mu[0].detach().cpu(), 3)
    path = cfg.paths.output_dir / "vae_reconstruction.png"
    _save_vae_reconstruction(
        image.cpu(), latent_rgb, reconstruction, label, path
    )
    return path


@torch.no_grad()
def save_vae_latent_distribution(vae: VAE, loader, cfg: AppConfig) -> Path:
    """保存 VAE 潜变量均值、posterior 样本和标准差的分布图。"""

    check_visual_config(cfg)
    cfg.paths.output_dir.mkdir(parents=True, exist_ok=True)
    device = next(vae.parameters()).device
    images, _ = _collect_images(loader, cfg)
    mu, logvar = vae.encode(images.to(device))
    posterior = vae.reparameterize(mu, logvar)
    std = torch.exp(0.5 * logvar)
    path = cfg.paths.output_dir / "vae_latent_distribution.png"
    _save_vae_distribution(
        mu.detach().cpu().flatten(),
        posterior.detach().cpu().flatten(),
        std.detach().cpu().flatten(),
        path,
    )
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


def _feature_map_pca_rgb(feature_map: torch.Tensor, components: int) -> torch.Tensor:
    channels, height, width = feature_map.shape
    flat = feature_map.reshape(channels, -1).T
    centered = flat - flat.mean(dim=0, keepdim=True)
    q = min(components, centered.shape[0], centered.shape[1])
    _, _, v = torch.pca_lowrank(centered, q=q)
    projected = centered @ v[:, :q]
    rgb = projected[:, -3:].T.reshape(3, height, width)
    return torch.stack([_normalize_map(channel) for channel in rgb])


def _save_rgb_map(rgb_map: torch.Tensor, path: Path, title: str) -> None:
    plt = _matplotlib()
    fig, axis = plt.subplots(figsize=(4, 4))
    axis.imshow(rgb_map.permute(1, 2, 0).numpy())
    axis.set_title(title)
    axis.axis("off")
    fig.tight_layout(pad=0.1)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _normalize_map(feature_map: torch.Tensor) -> torch.Tensor:
    min_value = feature_map.min()
    max_value = feature_map.max()
    return (feature_map - min_value) / (max_value - min_value + 1e-6)


def _save_vae_reconstruction(
    image: torch.Tensor,
    latent_rgb: torch.Tensor,
    reconstruction: torch.Tensor,
    label: int,
    path: Path,
) -> None:
    plt = _matplotlib()
    error = (image - reconstruction).abs()
    fig, axes = plt.subplots(1, 4, figsize=(8, 2.4))
    panels = [
        (image[0, 0], "MNIST input", "gray"),
        (latent_rgb.permute(1, 2, 0), "Latent PCA 32x14x14", None),
        (reconstruction[0, 0], "VAE reconstruction", "gray"),
        (error[0, 0], "Absolute error", "magma"),
    ]
    for axis, (panel, title, cmap) in zip(axes, panels):
        axis.imshow(panel.numpy(), cmap=cmap, vmin=0 if cmap == "gray" else None, vmax=1 if cmap == "gray" else None)
        axis.set_title(title, fontsize=8)
        axis.axis("off")
    fig.suptitle(f"VAE compression and reconstruction | label={label}", fontsize=10)
    fig.tight_layout(pad=0.2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_vae_distribution(
    mu: torch.Tensor, posterior: torch.Tensor, std: torch.Tensor, path: Path
) -> None:
    plt = _matplotlib()
    xs = torch.linspace(-4, 4, 200)
    normal = torch.exp(-0.5 * xs.pow(2)) / (2 * torch.pi) ** 0.5
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].hist(mu.numpy(), bins=80, density=True, alpha=0.55, label="mu")
    axes[0].hist(posterior.numpy(), bins=80, density=True, alpha=0.45, label="posterior sample")
    axes[0].plot(xs.numpy(), normal.numpy(), color="black", linewidth=1, label="N(0,1)")
    axes[0].set_title("Latent value distribution")
    axes[0].legend(fontsize=7)
    axes[1].hist(std.numpy(), bins=80, density=True, alpha=0.7, color="tab:green")
    axes[1].set_title("Posterior std distribution")
    fig.tight_layout()
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
