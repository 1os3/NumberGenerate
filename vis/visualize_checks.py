from config.schema import AppConfig


def check_visualization_checkpoints(cfg: AppConfig) -> None:
    # 校验对象: run_visualization 的 cfg.paths.vae_checkpoint/flow_checkpoint —— 可视化前必须已有训练权重
    missing_paths = [
        path
        for path in [cfg.paths.vae_checkpoint, cfg.paths.flow_checkpoint]
        if not path.exists()
    ]
    if missing_paths:
        joined = "、".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            f"缺少可视化所需 checkpoint: {joined}。请先运行 "
            "python -m train.vae_trainer 和 python -m train.flow_trainer。"
        )
