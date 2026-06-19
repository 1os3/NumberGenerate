from pathlib import Path


def check_config_path(path: Path) -> None:
    # 校验对象: load_config 的入参 config_path —— 配置文件必须存在且为 YAML
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("配置文件必须使用 .yaml 或 .yml 后缀。")
