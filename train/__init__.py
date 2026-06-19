"""提供训练与采样入口。"""

from importlib import import_module


_EXPORTS = {
    "flow_matching_loss": ("train.flow_trainer", "flow_matching_loss"),
    "sample_flow_time": ("train.flow_trainer", "sample_flow_time"),
    "sample_vae_posterior": ("train.flow_trainer", "sample_vae_posterior"),
    "sample_flow": ("train.sampling", "sample_flow"),
    "train_flow": ("train.flow_trainer", "train_flow"),
    "train_vae": ("train.vae_trainer", "train_vae"),
    "vae_loss": ("train.vae_trainer", "vae_loss"),
}

__all__ = [
    "flow_matching_loss",
    "sample_flow_time",
    "sample_vae_posterior",
    "sample_flow",
    "train_flow",
    "train_vae",
    "vae_loss",
]


def __getattr__(name: str):
    """按需导入公开训练接口，避免 python -m 执行子模块时提前加载。"""

    if name not in _EXPORTS:
        raise AttributeError(f"module 'train' has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """让交互式补全能看到惰性导出的公开接口。"""

    return sorted([*globals(), *__all__])
