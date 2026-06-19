"""训练数字条件潜空间 Flow Matching 模型。

模块: train/flow_trainer.py
依赖: argparse, torch, torch.nn.functional, config, data.mnist, model, train.common
读取配置: train.flow_epochs, train.flow_lr, train.weight_decay, train.grad_clip, train.log_interval, train.max_train_steps, sample.sampling_steps, paths.vae_checkpoint, paths.flow_checkpoint
对外接口:
    - train_flow(cfg) -> dict
    - flow_matching_loss(pred_velocity, target_velocity) -> Tensor
说明: Flow 训练冻结 VAE，只在潜空间学习从噪声到数据潜变量的速度场。
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from config import AppConfig, load_config
from data import get_mnist_loaders
from model import FlowModel, VAE
from train.common import load_checkpoint, prepare_runtime, save_checkpoint
from train.flow_trainer_checks import check_flow_batch, check_flow_training_config


def train_flow(cfg: AppConfig) -> dict:
    """训练 Flow Matching 模型并保存 checkpoint。

    参数:
        cfg: 项目配置对象，读取 Flow 训练超参和 checkpoint 路径
    返回:
        包含最终 epoch、step 和 loss 的指标字典
    """

    check_flow_training_config(cfg)
    device = prepare_runtime(cfg)
    train_loader, _ = get_mnist_loaders(cfg)
    vae = _load_frozen_vae(cfg, device)
    flow = FlowModel(cfg).to(device)
    optimizer = torch.optim.AdamW(
        flow.parameters(), lr=cfg.train.flow_lr, weight_decay=cfg.train.weight_decay
    )
    metrics = {"epoch": 0, "step": 0, "loss": float("nan")}
    for epoch in range(1, cfg.train.flow_epochs + 1):
        metrics = _train_flow_epoch(
            flow, vae, train_loader, optimizer, cfg, device, epoch, metrics["step"]
        )
        if cfg.train.max_train_steps is not None and metrics["step"] >= cfg.train.max_train_steps:
            break
    save_checkpoint(
        cfg.paths.flow_checkpoint,
        {"model": flow.state_dict(), "optimizer": optimizer.state_dict(), "metrics": metrics},
    )
    return metrics


def flow_matching_loss(
    pred_velocity: torch.Tensor, target_velocity: torch.Tensor
) -> torch.Tensor:
    """计算预测速度与目标速度的均方误差。"""

    return F.mse_loss(pred_velocity, target_velocity)


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser(description="训练 MNIST 潜空间 Flow Matching 模型")
    parser.add_argument("--config", type=str, default=None, help="可选 YAML 覆盖配置")
    args = parser.parse_args()
    metrics = train_flow(load_config(args.config))
    print(metrics)


def _load_frozen_vae(cfg: AppConfig, device: torch.device) -> VAE:
    vae = VAE(cfg).to(device)
    state = load_checkpoint(cfg.paths.vae_checkpoint, device)
    vae.load_state_dict(state["model"])
    vae.eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    return vae


def _train_flow_epoch(
    flow: FlowModel,
    vae: VAE,
    train_loader,
    optimizer: torch.optim.Optimizer,
    cfg: AppConfig,
    device: torch.device,
    epoch: int,
    start_step: int,
) -> dict:
    flow.train()
    running_loss = 0.0
    last_global_step = start_step
    local_step = 0
    for step, (images, labels) in enumerate(train_loader, start=1):
        global_step = start_step + step
        check_flow_batch(images, labels, cfg)
        images = images.to(device)
        labels = labels.to(device, dtype=torch.long)
        with torch.no_grad():
            z_1, _ = vae.encode(images)
        z_0 = torch.randn_like(z_1)
        time_index = torch.randint(0, cfg.sample.sampling_steps, (z_1.shape[0],), device=device)
        t = time_index.float() / cfg.sample.sampling_steps
        view_t = t.view(-1, 1, 1, 1)
        z_t = (1 - view_t) * z_0 + view_t * z_1
        target_velocity = z_1 - z_0
        optimizer.zero_grad(set_to_none=True)
        pred_velocity = flow(z_t, t, labels)
        loss = flow_matching_loss(pred_velocity, target_velocity)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), cfg.train.grad_clip)
        optimizer.step()
        running_loss += loss.item()
        local_step = step
        last_global_step = global_step
        if global_step % cfg.train.log_interval == 0:
            print({"epoch": epoch, "step": global_step, "loss": running_loss / step})
        if cfg.train.max_train_steps is not None and global_step >= cfg.train.max_train_steps:
            break
    return {"epoch": epoch, "step": last_global_step, "loss": running_loss / max(1, local_step)}


if __name__ == "__main__":
    main()
