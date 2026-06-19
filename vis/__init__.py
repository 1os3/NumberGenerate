"""提供项目可视化入口。"""

from importlib import import_module


_EXPORTS = {
    "run_visualization": ("vis.visualize", "run_visualization"),
    "save_flow_feature_pca": ("vis.plots", "save_flow_feature_pca"),
    "save_generation_steps": ("vis.plots", "save_generation_steps"),
    "save_vae_latent_pca": ("vis.plots", "save_vae_latent_pca"),
}

__all__ = [
    "run_visualization",
    "save_flow_feature_pca",
    "save_generation_steps",
    "save_vae_latent_pca",
]


def __getattr__(name: str):
    """按需导入公开可视化接口，避免执行子模块时被包初始化提前加载。"""

    if name not in _EXPORTS:
        raise AttributeError(f"module 'vis' has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """让交互式补全能看到惰性导出的公开接口。"""

    return sorted([*globals(), *__all__])
