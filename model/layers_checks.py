import torch


def check_positive_channels(channels: int, label: str) -> None:
    # 校验对象: 模型层构造参数 channels —— 通道数必须为正
    if channels <= 0:
        raise ValueError(f"{label} 必须为正整数。")


def check_expansion_ratio(expansion_ratio: int) -> None:
    # 校验对象: DepthwiseSeparableBlock 的 expansion_ratio —— 扩展倍率必须为正
    if expansion_ratio <= 0:
        raise ValueError("expansion_ratio 必须为正整数。")


def check_layer_tensor(x: torch.Tensor, channels: int, label: str) -> None:
    # 校验对象: 2D 卷积层输入 x —— 必须是指定通道数的四维张量
    if x.ndim != 4:
        raise ValueError(f"{label} 必须是形状 [N,C,H,W] 的四维张量。")
    if x.shape[1] != channels:
        raise ValueError(f"{label} 的通道数应为 {channels}，实际为 {x.shape[1]}。")


def check_condition_tensor(cond: torch.Tensor, batch: int, condition_dim: int) -> None:
    # 校验对象: AdaLN2d 的入参 cond —— 条件向量必须与 batch 和条件维度匹配
    if cond.ndim != 2:
        raise ValueError("cond 必须是形状 [N,D] 的二维张量。")
    if cond.shape != (batch, condition_dim):
        raise ValueError(f"cond 期望形状 {(batch, condition_dim)}，实际为 {tuple(cond.shape)}。")
