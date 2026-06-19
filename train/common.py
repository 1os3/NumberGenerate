"""提供训练脚本共用的随机种子、设备和 checkpoint 工具。

模块: train/common.py
依赖: random, numpy, torch, config.schema, train.common_checks
读取配置: train.seed, train.device, paths.checkpoint_dir, paths.output_dir, paths.log_dir
对外接口:
    - prepare_runtime(cfg) -> torch.device
    - save_checkpoint(path, payload) -> None
    - load_checkpoint(path, device) -> dict
    - empty_training_metrics() -> dict
    - load_latest_training_state(path, model, optimizer, device, label) -> dict
说明: 运行目录只在项目目录内创建，避免训练产物散落到外部。
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from config.schema import AppConfig
from train.common_checks import (
    check_checkpoint_metrics,
    check_checkpoint_path,
    check_runtime_paths,
    check_training_checkpoint_state,
)


def prepare_runtime(cfg: AppConfig) -> torch.device:
    """创建运行目录、设置随机种子并返回训练设备。"""

    check_runtime_paths(cfg)
    cfg.paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.log_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(cfg.train.seed)
    return _resolve_device(cfg.train.device)


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """保存训练状态到指定 checkpoint 路径。"""

    check_checkpoint_path(path, "checkpoint")
    torch.save(payload, path)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """从 checkpoint 文件读取训练状态。"""

    if not path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {path}")
    state = torch.load(path, map_location=device)
    if not isinstance(state, dict):
        raise ValueError("checkpoint 内容必须是字典。")
    return state


def empty_training_metrics() -> dict[str, Any]:
    """创建从头训练时使用的空指标字典。"""

    return {"epoch": 0, "step": 0, "loss": float("nan")}


def load_latest_training_state(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    label: str,
) -> dict[str, Any]:
    """如果最新 checkpoint 存在，则恢复模型、优化器和训练指标。

    参数:
        path: 单文件 checkpoint 路径
        model: 需要恢复参数的模型
        optimizer: 需要恢复动量等状态的优化器
        device: checkpoint 加载设备
        label: 错误信息中使用的模型名称
    返回:
        恢复后的指标；若文件不存在则返回从头训练指标
    """

    if not path.exists():
        print(f"未发现 {label} checkpoint，将从头训练: {path}")
        return empty_training_metrics()
    state = load_checkpoint(path, device)
    check_training_checkpoint_state(state, label)
    try:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
    except (RuntimeError, ValueError) as exc:
        raise RuntimeError(
            f"{label} checkpoint 与当前模型或优化器结构不兼容: {path}。"
            "如果需要重新训练，请删除该 checkpoint。"
        ) from exc
    metrics = state["metrics"]
    check_checkpoint_metrics(metrics, label)
    restored = {
        "epoch": int(metrics["epoch"]),
        "step": int(metrics["step"]),
        "loss": float(metrics["loss"]),
    }
    print(
        f"已恢复 {label} checkpoint: {path}，"
        f"下一轮从 epoch {restored['epoch'] + 1} 开始。"
    )
    return restored


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("配置要求 CUDA，但当前环境不可用。")
    return torch.device(device_name)
