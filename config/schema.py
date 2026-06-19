"""定义项目配置结构并在加载期完成参数校验。

模块: config/schema.py
依赖: pathlib, dataclasses
读取配置: 无
对外接口:
    - AppConfig: 全项目配置对象
    - build_config(raw, project_root) -> AppConfig
说明: 配置值来自 YAML，本文件只定义类型、路径解析和合法性约束。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PathsConfig:
    """保存所有项目内读写路径。"""

    data_dir: Path
    checkpoint_dir: Path
    output_dir: Path
    log_dir: Path
    vae_checkpoint: Path
    flow_checkpoint: Path


@dataclass(frozen=True)
class DataConfig:
    """保存 MNIST 数据加载参数。"""

    batch_size: int
    num_workers: int
    download: bool
    pin_memory: bool


@dataclass(frozen=True)
class TrainConfig:
    """保存训练循环使用的超参数。"""

    seed: int
    device: str
    vae_epochs: int
    flow_epochs: int
    vae_lr: float
    flow_lr: float
    weight_decay: float
    vae_beta: float
    grad_clip: float
    log_interval: int
    max_train_steps: int | None


@dataclass(frozen=True)
class ModelConfig:
    """保存 VAE 与 Flow 的结构参数。"""

    image_channels: int
    image_size: int
    num_classes: int
    vae_hidden_channels: int
    latent_channels: int
    latent_size: int
    flow_hidden_channels: int
    flow_depth: int
    convnext_expansion: int
    time_frequencies: int
    condition_dim: int


@dataclass(frozen=True)
class SampleConfig:
    """保存生成采样参数。"""

    sampling_steps: int
    history_steps: int
    num_samples_per_digit: int


@dataclass(frozen=True)
class VisualConfig:
    """保存可视化参数。"""

    pca_samples: int
    grid_columns: int


@dataclass(frozen=True)
class AppConfig:
    """聚合项目所有配置分组。"""

    project_root: Path
    paths: PathsConfig
    data: DataConfig
    train: TrainConfig
    model: ModelConfig
    sample: SampleConfig
    visual: VisualConfig


def build_config(raw: dict[str, Any], project_root: Path) -> AppConfig:
    """从字典构造强类型配置对象。

    参数:
        raw: YAML 读取后的嵌套字典
        project_root: 项目根目录路径
    返回:
        AppConfig 配置对象
    """

    _check_mapping(raw, "root")
    root = project_root.resolve()
    cfg = AppConfig(
        project_root=root,
        paths=_build_paths(_section(raw, "paths"), root),
        data=_build_data(_section(raw, "data")),
        train=_build_train(_section(raw, "train")),
        model=_build_model(_section(raw, "model")),
        sample=_build_sample(_section(raw, "sample")),
        visual=_build_visual(_section(raw, "visual")),
    )
    _check_config_relations(cfg)
    return cfg


def _build_paths(raw: dict[str, Any], project_root: Path) -> PathsConfig:
    return PathsConfig(
        data_dir=_project_path(_text(raw, "data_dir"), project_root, "paths.data_dir"),
        checkpoint_dir=_project_path(
            _text(raw, "checkpoint_dir"), project_root, "paths.checkpoint_dir"
        ),
        output_dir=_project_path(_text(raw, "output_dir"), project_root, "paths.output_dir"),
        log_dir=_project_path(_text(raw, "log_dir"), project_root, "paths.log_dir"),
        vae_checkpoint=_project_path(
            _text(raw, "vae_checkpoint"), project_root, "paths.vae_checkpoint"
        ),
        flow_checkpoint=_project_path(
            _text(raw, "flow_checkpoint"), project_root, "paths.flow_checkpoint"
        ),
    )


def _build_data(raw: dict[str, Any]) -> DataConfig:
    return DataConfig(
        batch_size=_positive_int(raw, "batch_size", "data.batch_size"),
        num_workers=_non_negative_int(raw, "num_workers", "data.num_workers"),
        download=_boolean(raw, "download", "data.download"),
        pin_memory=_boolean(raw, "pin_memory", "data.pin_memory"),
    )


def _build_train(raw: dict[str, Any]) -> TrainConfig:
    return TrainConfig(
        seed=_integer(raw, "seed", "train.seed"),
        device=_device(_text(raw, "device"), "train.device"),
        vae_epochs=_positive_int(raw, "vae_epochs", "train.vae_epochs"),
        flow_epochs=_positive_int(raw, "flow_epochs", "train.flow_epochs"),
        vae_lr=_positive_float(raw, "vae_lr", "train.vae_lr"),
        flow_lr=_positive_float(raw, "flow_lr", "train.flow_lr"),
        weight_decay=_non_negative_float(raw, "weight_decay", "train.weight_decay"),
        vae_beta=_non_negative_float(raw, "vae_beta", "train.vae_beta"),
        grad_clip=_positive_float(raw, "grad_clip", "train.grad_clip"),
        log_interval=_positive_int(raw, "log_interval", "train.log_interval"),
        max_train_steps=_optional_positive_int(
            raw, "max_train_steps", "train.max_train_steps"
        ),
    )


def _build_model(raw: dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        image_channels=_positive_int(raw, "image_channels", "model.image_channels"),
        image_size=_positive_int(raw, "image_size", "model.image_size"),
        num_classes=_positive_int(raw, "num_classes", "model.num_classes"),
        vae_hidden_channels=_positive_int(
            raw, "vae_hidden_channels", "model.vae_hidden_channels"
        ),
        latent_channels=_positive_int(raw, "latent_channels", "model.latent_channels"),
        latent_size=_positive_int(raw, "latent_size", "model.latent_size"),
        flow_hidden_channels=_positive_int(
            raw, "flow_hidden_channels", "model.flow_hidden_channels"
        ),
        flow_depth=_positive_int(raw, "flow_depth", "model.flow_depth"),
        convnext_expansion=_positive_int(
            raw, "convnext_expansion", "model.convnext_expansion"
        ),
        time_frequencies=_positive_int(
            raw, "time_frequencies", "model.time_frequencies"
        ),
        condition_dim=_positive_int(raw, "condition_dim", "model.condition_dim"),
    )


def _build_sample(raw: dict[str, Any]) -> SampleConfig:
    return SampleConfig(
        sampling_steps=_positive_int(raw, "sampling_steps", "sample.sampling_steps"),
        history_steps=_positive_int(raw, "history_steps", "sample.history_steps"),
        num_samples_per_digit=_positive_int(
            raw, "num_samples_per_digit", "sample.num_samples_per_digit"
        ),
    )


def _build_visual(raw: dict[str, Any]) -> VisualConfig:
    return VisualConfig(
        pca_samples=_positive_int(raw, "pca_samples", "visual.pca_samples"),
        grid_columns=_positive_int(raw, "grid_columns", "visual.grid_columns"),
    )


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    _check_mapping(value, key)
    return value


def _check_mapping(value: Any, name: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{name} 必须是字典配置段。")


def _text(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} 必须是非空字符串。")
    return value


def _integer(raw: dict[str, Any], key: str, label: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} 必须是整数。")
    return value


def _positive_int(raw: dict[str, Any], key: str, label: str) -> int:
    value = _integer(raw, key, label)
    if value <= 0:
        raise ValueError(f"{label} 必须 > 0。")
    return value


def _non_negative_int(raw: dict[str, Any], key: str, label: str) -> int:
    value = _integer(raw, key, label)
    if value < 0:
        raise ValueError(f"{label} 必须 >= 0。")
    return value


def _optional_positive_int(raw: dict[str, Any], key: str, label: str) -> int | None:
    if raw.get(key) is None:
        return None
    return _positive_int(raw, key, label)


def _positive_float(raw: dict[str, Any], key: str, label: str) -> float:
    value = _number(raw, key, label)
    if value <= 0:
        raise ValueError(f"{label} 必须 > 0。")
    return value


def _non_negative_float(raw: dict[str, Any], key: str, label: str) -> float:
    value = _number(raw, key, label)
    if value < 0:
        raise ValueError(f"{label} 必须 >= 0。")
    return value


def _number(raw: dict[str, Any], key: str, label: str) -> float:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} 必须是数字。")
    return float(value)


def _boolean(raw: dict[str, Any], key: str, label: str) -> bool:
    value = raw.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{label} 必须是布尔值。")
    return value


def _device(value: str, label: str) -> str:
    if value == "auto" or value == "cpu" or value.startswith("cuda"):
        return value
    raise ValueError(f"{label} 只能是 auto、cpu 或 cuda 设备字符串。")


def _project_path(value: str, project_root: Path, label: str) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (project_root / path).resolve()
    if not resolved.is_relative_to(project_root):
        raise ValueError(f"{label} 必须位于项目目录内。")
    return resolved


def _check_config_relations(cfg: AppConfig) -> None:
    expected_latent_size = cfg.model.image_size // 2
    if cfg.model.image_size % 2 != 0:
        raise ValueError("model.image_size 必须能被 2 整除。")
    if cfg.model.latent_size != expected_latent_size:
        raise ValueError("model.latent_size 必须等于 model.image_size // 2。")
    if cfg.sample.history_steps > cfg.sample.sampling_steps:
        raise ValueError("sample.history_steps 不得大于 sample.sampling_steps。")
