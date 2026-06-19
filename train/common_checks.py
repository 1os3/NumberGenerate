"""校验训练运行环境、checkpoint 路径和断点续训状态。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config.schema import AppConfig


def check_runtime_paths(cfg: AppConfig) -> None:
    # 校验对象: 训练输出路径 cfg.paths —— 所有输出路径必须位于项目目录内
    paths = [
        cfg.paths.checkpoint_dir,
        cfg.paths.output_dir,
        cfg.paths.log_dir,
        cfg.paths.vae_checkpoint,
        cfg.paths.flow_checkpoint,
    ]
    if any(not path.is_relative_to(cfg.project_root) for path in paths):
        raise ValueError("训练输出路径必须位于项目目录内。")


def check_checkpoint_path(path: Path, label: str) -> None:
    # 校验对象: checkpoint 路径参数 —— checkpoint 必须位于已有父目录下
    if not path.parent.exists():
        raise FileNotFoundError(f"{label} 的父目录不存在: {path.parent}")


def check_training_checkpoint_state(state: Any, label: str) -> None:
    # 校验对象: 训练 checkpoint 字典 —— 断点续训必须同时包含模型、优化器和指标
    if not isinstance(state, dict):
        raise ValueError(f"{label} checkpoint 必须是字典。")
    missing_keys = [
        key for key in ["model", "optimizer", "metrics"] if key not in state
    ]
    if missing_keys:
        joined = ", ".join(missing_keys)
        raise KeyError(f"{label} checkpoint 缺少字段: {joined}。")


def check_checkpoint_metrics(metrics: Any, label: str) -> None:
    # 校验对象: checkpoint 中的 metrics 字典 —— epoch/step 用于决定续训起点
    if not isinstance(metrics, dict):
        raise ValueError(f"{label} checkpoint metrics 必须是字典。")
    epoch = metrics.get("epoch")
    step = metrics.get("step")
    loss = metrics.get("loss")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError(f"{label} checkpoint metrics.epoch 必须是非负整数。")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(f"{label} checkpoint metrics.step 必须是非负整数。")
    if isinstance(loss, bool) or not isinstance(loss, (int, float)):
        raise ValueError(f"{label} checkpoint metrics.loss 必须是数字。")
