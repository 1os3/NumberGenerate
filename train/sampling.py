"""提供 Flow Matching 生成采样函数。

模块: train/sampling.py
依赖: torch, config.schema, model.flow, model.vae, train.sampling_checks
读取配置: model.latent_channels, model.latent_size, sample.sampling_steps, sample.history_steps
对外接口:
    - sample_flow(flow, vae, labels, cfg, return_history=False) -> Tensor | tuple[Tensor, Tensor]
说明: 采样从高斯噪声潜变量出发，Euler 积分后用 VAE decoder 还原图像。
"""

from __future__ import annotations

import torch

from config.schema import AppConfig
from model import FlowModel, VAE
from train.sampling_checks import check_sample_inputs


@torch.no_grad()
def sample_flow(
    flow: FlowModel,
    vae: VAE,
    labels: torch.Tensor,
    cfg: AppConfig,
    return_history: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """根据数字标签生成 MNIST 图像。

    参数:
        flow: 已训练的 FlowModel
        vae: 已训练的 VAE
        labels: 形状 [N] 的数字标签
        cfg: 项目配置对象，读取潜空间尺寸与采样步数
        return_history: 是否返回逐步生成历史
    返回:
        生成图像，或生成图像与历史图像序列
    """

    check_sample_inputs(labels, cfg)
    device = next(flow.parameters()).device
    labels = labels.to(device)
    z = torch.randn(
        labels.shape[0],
        cfg.model.latent_channels,
        cfg.model.latent_size,
        cfg.model.latent_size,
        device=device,
    )
    flow.eval()
    vae.eval()
    history = []
    step_size = 1.0 / cfg.sample.sampling_steps
    history_interval = max(1, cfg.sample.sampling_steps // cfg.sample.history_steps)
    for step in range(cfg.sample.sampling_steps):
        t = torch.full((labels.shape[0],), step * step_size, device=device)
        z = z + step_size * flow(z, t, labels)
        if return_history and ((step + 1) % history_interval == 0):
            history.append(vae.decode(z).detach().cpu())
    images = vae.decode(z).clamp(0, 1).detach().cpu()
    if not return_history:
        return images
    return images, torch.stack(history).clamp(0, 1)
