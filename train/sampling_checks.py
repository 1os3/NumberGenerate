import torch

from config.schema import AppConfig


def check_sample_inputs(labels: torch.Tensor, cfg: AppConfig) -> None:
    # 校验对象: sample_flow 的入参 labels —— 标签必须是一维 long 张量且位于类别范围内
    if labels.ndim != 1:
        raise ValueError("labels 必须是一维张量。")
    if labels.dtype != torch.long:
        raise ValueError("labels 必须是 torch.long 类型。")
    if labels.numel() and (labels.min() < 0 or labels.max() >= cfg.model.num_classes):
        raise ValueError("labels 必须位于 [0, num_classes) 范围内。")
