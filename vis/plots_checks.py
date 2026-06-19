import torch

from config.schema import AppConfig


def check_visual_config(cfg: AppConfig) -> None:
    # 校验对象: 可视化函数入参 cfg —— 输出目录必须位于项目目录内
    if not cfg.paths.output_dir.is_relative_to(cfg.project_root):
        raise ValueError("paths.output_dir 必须位于项目目录内。")


def check_visual_images(images: torch.Tensor) -> None:
    # 校验对象: 图像网格绘制输入 images —— 必须是 [N,1,H,W] 张量
    if images.ndim != 4 or images.shape[1] != 1:
        raise ValueError("images 必须是形状 [N,1,H,W] 的灰度图张量。")
