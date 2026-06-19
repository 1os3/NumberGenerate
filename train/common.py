"""提供训练脚本共用的随机种子、设备和 checkpoint 工具。

模块: train/common.py
依赖: random, numpy, torch, config.schema, train.common_checks
读取配置: train.seed, train.device, paths.checkpoint_dir, paths.output_dir, paths.log_dir
对外接口:
    - prepare_runtime(cfg) -> torch.device
    - save_checkpoint(path, payload) -> None
    - load_checkpoint(path, device) -> dict
说明: 运行目录只在项目目录内创建，避免训练产物散落到外部。
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from config.schema import AppConfig
from train.common_checks import check_checkpoint_path, check_runtime_paths


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
