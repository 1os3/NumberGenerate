"""提供项目配置加载入口。"""

from config.loader import load_config
from config.schema import (
    AppConfig,
    DataConfig,
    ModelConfig,
    PathsConfig,
    SampleConfig,
    TrainConfig,
    VisualConfig,
)

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
