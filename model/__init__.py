"""提供模型层公开入口。"""

from importlib import import_module


_EXPORTS = {
    "AdaLN2d": ("model.layers", "AdaLN2d"),
    "ConditionalDepthwiseSeparableBlock": (
        "model.layers",
        "ConditionalDepthwiseSeparableBlock",
    ),
    "DepthwiseSeparableBlock": ("model.layers", "DepthwiseSeparableBlock"),
    "FlowModel": ("model.flow", "FlowModel"),
    "LayerNorm2d": ("model.layers", "LayerNorm2d"),
    "ResidualBlock": ("model.layers", "ResidualBlock"),
    "VAE": ("model.vae", "VAE"),
}

__all__ = [
    "AdaLN2d",
    "ConditionalDepthwiseSeparableBlock",
    "DepthwiseSeparableBlock",
    "FlowModel",
    "LayerNorm2d",
    "ResidualBlock",
    "VAE",
]


def __getattr__(name: str):
    """按需导入公开模型接口，避免执行子模块时被包初始化提前加载。"""

    if name not in _EXPORTS:
        raise AttributeError(f"module 'model' has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """让交互式补全能看到惰性导出的公开接口。"""

    return sorted([*globals(), *__all__])
