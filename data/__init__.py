"""提供数据加载入口。"""

from importlib import import_module


_EXPORTS = {
    "get_mnist_loaders": ("data.mnist", "get_mnist_loaders"),
}

__all__ = ["get_mnist_loaders"]


def __getattr__(name: str):
    """按需导入公开数据接口，避免执行子模块时被包初始化提前加载。"""

    if name not in _EXPORTS:
        raise AttributeError(f"module 'data' has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """让交互式补全能看到惰性导出的公开接口。"""

    return sorted([*globals(), *__all__])
