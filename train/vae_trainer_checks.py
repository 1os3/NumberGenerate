import torch

from config.schema import AppConfig


def check_vae_training_config(cfg: AppConfig) -> None:
    # 校验对象: train_vae 的入参 cfg —— VAE checkpoint 必须位于 checkpoint 目录下
    if cfg.paths.vae_checkpoint.parent != cfg.paths.checkpoint_dir:
        raise ValueError("paths.vae_checkpoint 必须位于 paths.checkpoint_dir 下。")


def check_vae_batch(images: torch.Tensor, cfg: AppConfig) -> None:
    # 校验对象: VAE 训练 batch images —— 图像 batch 必须匹配配置尺寸
    expected = (cfg.model.image_channels, cfg.model.image_size, cfg.model.image_size)
    if images.ndim != 4 or tuple(images.shape[1:]) != expected:
        raise ValueError(f"images 期望形状 [N,{expected[0]},{expected[1]},{expected[2]}]。")
