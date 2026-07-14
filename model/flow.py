"""定义数字条件潜空间 Flow Matching 模型。

模块: model/flow.py
依赖: torch, torch.nn, config.schema, model.layers, model.flow_checks
读取配置: model.latent_channels, model.latent_size, model.num_classes, model.flow_hidden_channels, model.flow_depth, model.convnext_expansion, model.time_frequencies, model.condition_dim
对外接口:
    - FlowModel(cfg) -> nn.Module
    - FlowModel.forward(z_t, t, labels) -> Tensor
    - FlowModel.predict_with_features(z_t, t, labels) -> tuple[Tensor, list[Tensor]]
    - FlowModel.extract_features(z_t, t, labels) -> list[Tensor]
说明: 只有 8 层条件 ConvNeXt 主干块使用 AdaLN2d，输入和输出投影保持普通卷积。
"""

from __future__ import annotations

import torch
from torch import nn

from config.schema import AppConfig
from model.flow_checks import check_flow_config, check_flow_inputs
from model.layers import ConditionalDepthwiseSeparableBlock, LayerNorm2d


class TimeEmbedding(nn.Module):
    """生成连续时间步的 sin/cos 频率嵌入。"""

    def __init__(self, frequencies: int) -> None:
        super().__init__()
        positions = torch.arange(frequencies, dtype=torch.float32)
        denominator = max(1, frequencies * 2)
        self.register_buffer("omegas", torch.pow(torch.tensor(100.0), -2 * positions / denominator))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """将 [N] 或 [N,1] 时间张量映射为 [N,2F]。"""

        flat_t = t.reshape(-1, 1).to(self.omegas.device, dtype=self.omegas.dtype)
        angles = flat_t * self.omegas.view(1, -1)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


class FlowModel(nn.Module):
    """在 VAE 潜空间预测线性 Flow Matching 速度场。"""

    def __init__(self, cfg: AppConfig) -> None:
        super().__init__()
        check_flow_config(cfg)
        hidden = cfg.model.flow_hidden_channels
        condition_dim = cfg.model.condition_dim
        self.cfg = cfg
        self.time_embedding = TimeEmbedding(cfg.model.time_frequencies)
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.model.time_frequencies * 2, condition_dim),
            nn.GELU(),
            nn.Linear(condition_dim, condition_dim),
        )
        self.label_embedding = nn.Embedding(cfg.model.num_classes, condition_dim)
        self.input_projection = nn.Conv2d(cfg.model.latent_channels, hidden, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                ConditionalDepthwiseSeparableBlock(
                    hidden, condition_dim, cfg.model.convnext_expansion
                )
                for _ in range(cfg.model.flow_depth)
            ]
        )
        self.output_norm = LayerNorm2d(hidden)
        self.output_projection = nn.Conv2d(hidden, cfg.model.latent_channels, kernel_size=1)

    def forward(
        self, z_t: torch.Tensor, t: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """预测从噪声潜变量指向数据潜变量的速度。"""

        velocity, _ = self.predict_with_features(z_t, t, labels)
        return velocity

    def predict_with_features(
        self, z_t: torch.Tensor, t: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """一次前向同时返回预测速度和各主干块输出特征。"""

        features = self.extract_features(z_t, t, labels)
        velocity = self.output_projection(self.output_norm(features[-1]))
        return velocity, features

    def extract_features(
        self, z_t: torch.Tensor, t: torch.Tensor, labels: torch.Tensor
    ) -> list[torch.Tensor]:
        """返回每个条件主干块后的中间特征。"""

        check_flow_inputs(z_t, t, labels, self.cfg)
        cond = self.time_mlp(self.time_embedding(t)) + self.label_embedding(labels)
        hidden = self.input_projection(z_t)
        features = []
        for block in self.blocks:
            hidden = block(hidden, cond)
            features.append(hidden)
        return features
