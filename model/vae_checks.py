import torch

from config.schema import AppConfig


def check_vae_config(cfg: AppConfig) -> None:
    # 校验对象: VAE 构造参数 cfg.model —— 潜空间尺寸必须对应 2 倍下采样
    if cfg.model.latent_size != cfg.model.image_size // 2:
        raise ValueError("VAE 期望 latent_size 等于 image_size // 2。")
    if cfg.model.vae_hidden_channels < 2:
        raise ValueError("model.vae_hidden_channels 必须不小于 2。")


def check_image_batch(x: torch.Tensor, cfg: AppConfig) -> None:
    # 校验对象: VAE.encode 的入参 x —— 图像 batch 必须匹配配置尺寸
    expected = (cfg.model.image_channels, cfg.model.image_size, cfg.model.image_size)
    if x.ndim != 4 or tuple(x.shape[1:]) != expected:
        raise ValueError(f"图像 batch 期望形状 [N,{expected[0]},{expected[1]},{expected[2]}]。")


def check_latent_batch(z: torch.Tensor, cfg: AppConfig) -> None:
    # 校验对象: VAE.decode 的入参 z —— 潜变量 batch 必须匹配配置尺寸
    expected = (cfg.model.latent_channels, cfg.model.latent_size, cfg.model.latent_size)
    if z.ndim != 4 or tuple(z.shape[1:]) != expected:
        raise ValueError(f"潜变量 batch 期望形状 [N,{expected[0]},{expected[1]},{expected[2]}]。")
