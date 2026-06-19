from config.schema import AppConfig


def check_mnist_config(cfg: AppConfig) -> None:
    # 校验对象: get_mnist_loaders 的入参 cfg —— 数据目录必须归属当前项目
    if not cfg.paths.data_dir.is_relative_to(cfg.project_root):
        raise ValueError("paths.data_dir 必须位于项目目录内。")
