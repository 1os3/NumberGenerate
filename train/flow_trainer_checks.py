import torch

from config.schema import AppConfig


def check_flow_training_config(cfg: AppConfig) -> None:
    # 校验对象: train_flow 的入参 cfg —— Flow checkpoint 必须位于 checkpoint 目录下
    if cfg.paths.flow_checkpoint.parent != cfg.paths.checkpoint_dir:
        raise ValueError("paths.flow_checkpoint 必须位于 paths.checkpoint_dir 下。")


def check_flow_batch(images: torch.Tensor, labels: torch.Tensor, cfg: AppConfig) -> None:
    # 校验对象: Flow 训练 batch images/labels —— 图像和标签 batch 必须对齐
    expected = (cfg.model.image_channels, cfg.model.image_size, cfg.model.image_size)
    if images.ndim != 4 or tuple(images.shape[1:]) != expected:
        raise ValueError(f"images 期望形状 [N,{expected[0]},{expected[1]},{expected[2]}]。")
    if labels.ndim != 1 or labels.shape[0] != images.shape[0]:
        raise ValueError("labels 必须是与 images batch 对齐的一维张量。")
