"""训练无条件 MNIST VAE。

模块: train/vae_trainer.py
依赖: argparse, torch, torch.nn.functional, config, data.mnist, model.vae, train.common
读取配置: train.vae_epochs, train.vae_lr, train.weight_decay, train.vae_kl_weight, train.grad_clip, train.grad_monitor_enabled, train.grad_small_threshold, train.grad_small_warn_ratio, train.log_interval, train.max_train_steps, paths.vae_checkpoint
对外接口:
    - train_vae(cfg) -> dict
    - vae_loss(recon, images, mu, logvar, cfg) -> Tensor
说明: VAE 训练只重建图像，不接收数字条件。
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from config import AppConfig, load_config
from data import get_mnist_loaders
from model import VAE
from train.common import (
    load_latest_training_state,
    prepare_runtime,
    save_checkpoint,
    summarize_gradients,
)
from train.vae_trainer_checks import check_vae_batch, check_vae_training_config


def train_vae(cfg: AppConfig) -> dict:
    """训练 VAE 并保存 checkpoint。

    参数:
        cfg: 项目配置对象，读取 VAE 训练超参和输出路径
    返回:
        包含最终 epoch、step 和 loss 的指标字典
    """

    check_vae_training_config(cfg)
    device = prepare_runtime(cfg)
    train_loader, _ = get_mnist_loaders(cfg)
    model = VAE(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.vae_lr, weight_decay=cfg.train.weight_decay
    )
    metrics = load_latest_training_state(
        cfg.paths.vae_checkpoint, model, optimizer, device, "VAE"
    )
    if cfg.train.max_train_steps is not None and metrics["step"] >= cfg.train.max_train_steps:
        print("VAE 已达到 train.max_train_steps，无需继续训练。")
        return metrics
    if metrics["epoch"] >= cfg.train.vae_epochs:
        print("VAE checkpoint 已达到配置的目标 epoch，无需继续训练。")
        return metrics
    for epoch in range(metrics["epoch"] + 1, cfg.train.vae_epochs + 1):
        metrics = _train_vae_epoch(
            model, train_loader, optimizer, cfg, device, epoch, metrics["step"]
        )
        save_checkpoint(
            cfg.paths.vae_checkpoint,
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "metrics": metrics,
            },
        )
        if cfg.train.max_train_steps is not None and metrics["step"] >= cfg.train.max_train_steps:
            break
    return metrics


def vae_loss(
    recon: torch.Tensor,
    images: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    cfg: AppConfig,
) -> torch.Tensor:
    """计算 VAE 二通道 BCE 重建损失与 KL 正则之和。"""

    recon_loss = F.binary_cross_entropy(recon, images)
    kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
    return recon_loss + cfg.train.vae_kl_weight * kl_loss


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser(description="训练 MNIST VAE")
    parser.add_argument("--config", type=str, default=None, help="可选 YAML 覆盖配置")
    args = parser.parse_args()
    metrics = train_vae(load_config(args.config))
    print(metrics)


def _train_vae_epoch(
    model: VAE,
    train_loader,
    optimizer: torch.optim.Optimizer,
    cfg: AppConfig,
    device: torch.device,
    epoch: int,
    start_step: int,
) -> dict:
    model.train()
    running_loss = 0.0
    last_global_step = start_step
    local_step = 0
    for step, (images, _) in enumerate(train_loader, start=1):
        global_step = start_step + step
        check_vae_batch(images, cfg)
        images = images.to(device)
        optimizer.zero_grad(set_to_none=True)
        recon, mu, logvar = model(images)
        loss = vae_loss(recon, images, mu, logvar, cfg)
        loss.backward()
        should_log = global_step % cfg.train.log_interval == 0
        grad_stats = None
        if cfg.train.grad_monitor_enabled and should_log:
            grad_stats = summarize_gradients(model, cfg.train.grad_small_threshold)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()
        running_loss += loss.item()
        local_step = step
        last_global_step = global_step
        if should_log:
            log = {"epoch": epoch, "step": global_step, "loss": running_loss / step}
            if grad_stats is not None:
                log.update(grad_stats)
            print(log)
            if (
                grad_stats is not None
                and grad_stats["grad_small_ratio"] >= cfg.train.grad_small_warn_ratio
            ):
                print(
                    {
                        "epoch": epoch,
                        "step": global_step,
                        "warning": "small gradients",
                        "threshold": cfg.train.grad_small_threshold,
                        "small_ratio": grad_stats["grad_small_ratio"],
                    }
                )
        if cfg.train.max_train_steps is not None and global_step >= cfg.train.max_train_steps:
            break
    return {"epoch": epoch, "step": last_global_step, "loss": running_loss / max(1, local_step)}


if __name__ == "__main__":
    main()
