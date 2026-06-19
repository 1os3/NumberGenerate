import torch

from config.schema import AppConfig


def check_flow_config(cfg: AppConfig) -> None:
    # 校验对象: FlowModel 构造参数 cfg.model —— 条件维度和主干深度必须支持 AdaLN 注入
    if cfg.model.condition_dim <= 0 or cfg.model.flow_depth <= 0:
        raise ValueError("Flow 条件维度和主干深度必须为正。")


def check_flow_inputs(
    z_t: torch.Tensor, t: torch.Tensor, labels: torch.Tensor, cfg: AppConfig
) -> None:
    # 校验对象: FlowModel.forward 的入参 z_t/t/labels —— batch、形状和标签范围必须匹配
    expected = (cfg.model.latent_channels, cfg.model.latent_size, cfg.model.latent_size)
    if z_t.ndim != 4 or tuple(z_t.shape[1:]) != expected:
        raise ValueError(f"z_t 期望形状 [N,{expected[0]},{expected[1]},{expected[2]}]。")
    if t.ndim not in {1, 2} or t.shape[0] != z_t.shape[0]:
        raise ValueError("t 必须是形状 [N] 或 [N,1] 的时间张量。")
    if labels.ndim != 1 or labels.shape[0] != z_t.shape[0]:
        raise ValueError("labels 必须是形状 [N] 的标签张量。")
    if labels.dtype != torch.long:
        raise ValueError("labels 必须是 torch.long 类型。")
    if labels.numel() and (labels.min() < 0 or labels.max() >= cfg.model.num_classes):
        raise ValueError("labels 必须位于 [0, num_classes) 范围内。")
