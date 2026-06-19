from pathlib import Path

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
