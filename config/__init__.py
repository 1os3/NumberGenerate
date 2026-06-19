"""提供项目配置加载入口。"""

from importlib import import_module


_EXPORTS = {
    "AppConfig": ("config.schema", "AppConfig"),
    "DataConfig": ("config.schema", "DataConfig"),
    "ModelConfig": ("config.schema", "ModelConfig"),
    "PathsConfig": ("config.schema", "PathsConfig"),
    "SampleConfig": ("config.schema", "SampleConfig"),
    "TrainConfig": ("config.schema", "TrainConfig"),
    "VisualConfig": ("config.schema", "VisualConfig"),
    "load_config": ("config.loader", "load_config"),
}

__all__ = [
    "AppConfig",
    "DataConfig",
    "ModelConfig",
    "PathsConfig",
    "SampleConfig",
    "TrainConfig",
    "VisualConfig",
    "load_config",
]


def __getattr__(name: str):
    """按需导入公开配置接口，避免执行子模块时被包初始化提前加载。"""

    if name not in _EXPORTS:
        raise AttributeError(f"module 'config' has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """让交互式补全能看到惰性导出的公开接口。"""

    return sorted([*globals(), *__all__])
