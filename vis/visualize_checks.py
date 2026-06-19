"""校验可视化入口所需的 checkpoint 文件。"""

from __future__ import annotations

from config.schema import AppConfig


VISUALIZATION_MODES = {"all", "vae"}


def check_visualization_mode(mode: str) -> None:
    # 校验对象: CLI/API 传入的可视化模式 —— 避免拼写错误导致加载逻辑不明确
    if mode not in VISUALIZATION_MODES:
        joined = ", ".join(sorted(VISUALIZATION_MODES))
        raise ValueError(f"可视化模式必须是以下之一: {joined}。")


def check_visualization_checkpoints(cfg: AppConfig, mode: str = "all") -> None:
    # 校验对象: run_visualization 的 checkpoint 路径 —— VAE-only 模式不要求 Flow 权重
    check_visualization_mode(mode)
    required_paths = [cfg.paths.vae_checkpoint]
    if mode == "all":
        required_paths.append(cfg.paths.flow_checkpoint)
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        joined = "、".join(str(path) for path in missing_paths)
        if mode == "vae":
            command = "python -m train.vae_trainer"
        else:
            command = "python -m train.vae_trainer 和 python -m train.flow_trainer"
        raise FileNotFoundError(
            f"缺少可视化所需 checkpoint: {joined}。请先运行 {command}。"
        )
