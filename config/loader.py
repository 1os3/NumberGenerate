"""加载 YAML 配置并返回强类型配置对象。

模块: config/loader.py
依赖: config.schema, config.loader_checks, pathlib
读取配置: config/default.yaml, 可选覆盖配置文件
对外接口:
    - load_config(config_path=None) -> AppConfig
说明: 覆盖配置只改写显式传入的字段，其余字段沿用默认配置。
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from config.loader_checks import check_config_path
from config.schema import AppConfig, build_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = Path(__file__).with_name("default.yaml")


def load_config(config_path: str | None = None) -> AppConfig:
    """读取默认配置和可选覆盖配置。

    参数:
        config_path: 相对项目根目录或绝对路径的 YAML 覆盖配置
    返回:
        AppConfig 配置对象
    """

    default_raw = _read_yaml(DEFAULT_CONFIG_PATH)
    if config_path is None:
        return build_config(default_raw, PROJECT_ROOT)

    override_path = _resolve_config_path(config_path)
    override_raw = _read_yaml(override_path)
    return build_config(_deep_merge(default_raw, override_raw), PROJECT_ROOT)


def _resolve_config_path(config_path: str) -> Path:
    path = Path(config_path)
    resolved = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    check_config_path(resolved)
    return resolved


def _read_yaml(path: Path) -> dict[str, Any]:
    check_config_path(path)
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("缺少 PyYAML，请先安装 requirements.txt 中的依赖。") from exc

    content = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(content, dict):
        raise ValueError(f"配置文件顶层必须是字典: {path}")
    return content


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
