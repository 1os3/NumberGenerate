"""验证默认配置、模型形状、条件注入和核心损失。

模块: tests/test_config_and_shapes.py
依赖: unittest, torch, config, model, train
读取配置: config/default.yaml
对外接口:
    - ConfigAndShapeTests(unittest.TestCase) -> None
说明: 测试不下载 MNIST，只验证可在随机张量上跑通的核心路径。
"""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

from config import load_config
from config.schema import build_config
from model import ConditionalDepthwiseSeparableBlock, FlowModel, LayerNorm2d, VAE
from train import flow_matching_loss, sample_flow, sample_vae_posterior, vae_loss


class ConfigAndShapeTests(unittest.TestCase):
    """覆盖配置加载、模型形状与损失反向传播。"""

    def setUp(self) -> None:
        self.cfg = _build_test_config()

    def test_default_config_loads(self) -> None:
        """默认配置应能加载出项目内路径。"""

        try:
            cfg = load_config()
        except RuntimeError as exc:
            if "PyYAML" in str(exc):
                self.skipTest("缺少 PyYAML，安装 requirements.txt 后会执行该测试。")
            raise
        self.assertTrue(cfg.paths.output_dir.is_relative_to(cfg.project_root))

    def test_vae_shapes_and_loss(self) -> None:
        """VAE encode/decode/forward 形状和损失应可反向传播。"""

        vae = VAE(self.cfg)
        self.assertIsInstance(vae.encoder.latent_norm, LayerNorm2d)
        images = torch.rand(2, 1, 28, 28)
        recon, mu, logvar = vae(images)
        self.assertEqual(tuple(mu.shape), (2, 32, 14, 14))
        self.assertEqual(tuple(recon.shape), tuple(images.shape))
        posterior_sample = sample_vae_posterior(vae, images)
        self.assertEqual(tuple(posterior_sample.shape), tuple(mu.shape))
        self.assertFalse(torch.equal(posterior_sample, mu))
        loss = vae_loss(recon, images, mu, logvar, self.cfg)
        loss.backward()
        self.assertEqual(loss.ndim, 0)

    def test_flow_shapes_condition_blocks_and_loss(self) -> None:
        """Flow 主干应使用条件块，输出速度形状应匹配潜变量。"""

        flow = FlowModel(self.cfg)
        self.assertIsInstance(flow.output_norm, LayerNorm2d)
        blocks_are_conditional = [
            isinstance(block, ConditionalDepthwiseSeparableBlock) for block in flow.blocks
        ]
        self.assertTrue(all(blocks_are_conditional))
        self.assertEqual(flow.input_projection.in_channels, 32)
        self.assertEqual(flow.input_projection.out_channels, 256)
        self.assertEqual(flow.output_projection.in_channels, 256)
        self.assertEqual(flow.output_projection.out_channels, 32)
        z_t = torch.randn(2, 32, 14, 14)
        t = torch.rand(2)
        labels = torch.tensor([0, 1], dtype=torch.long)
        pred = flow(z_t, t, labels)
        target = torch.randn_like(pred)
        self.assertEqual(tuple(pred.shape), tuple(z_t.shape))
        loss = flow_matching_loss(pred, target)
        loss.backward()
        self.assertEqual(loss.ndim, 0)

    def test_sample_flow_shape(self) -> None:
        """采样函数应输出 [N,1,28,28] 图像。"""

        vae = VAE(self.cfg)
        flow = FlowModel(self.cfg)
        labels = torch.tensor([0, 1], dtype=torch.long)
        images = sample_flow(flow, vae, labels, self.cfg)
        self.assertEqual(tuple(images.shape), (2, 1, 28, 28))


def _build_test_config():
    raw = {
        "paths": {
            "data_dir": "datasets",
            "checkpoint_dir": "checkpoints",
            "output_dir": "outputs",
            "log_dir": "logs",
            "vae_checkpoint": "checkpoints/vae.pt",
            "flow_checkpoint": "checkpoints/flow.pt",
        },
        "data": {
            "batch_size": 2,
            "num_workers": 0,
            "download": False,
            "pin_memory": False,
        },
        "train": {
            "seed": 42,
            "device": "cpu",
            "vae_epochs": 1,
            "flow_epochs": 1,
            "vae_lr": 0.001,
            "flow_lr": 0.0002,
            "weight_decay": 0.000001,
            "vae_kl_weight": 0.0001,
            "grad_clip": 1.0,
            "log_interval": 1,
            "max_train_steps": 1,
        },
        "model": {
            "image_channels": 1,
            "image_size": 28,
            "num_classes": 10,
            "vae_hidden_channels": 64,
            "latent_channels": 32,
            "latent_size": 14,
            "flow_hidden_channels": 256,
            "flow_depth": 8,
            "convnext_expansion": 2,
            "time_frequencies": 16,
            "condition_dim": 128,
        },
        "sample": {
            "sampling_steps": 2,
            "history_steps": 2,
            "num_samples_per_digit": 1,
        },
        "visual": {
            "pca_samples": 4,
            "grid_columns": 2,
            "feature_map_channels": 3,
            "feature_map_time": 0.5,
        },
    }
    return build_config(raw, Path.cwd())


if __name__ == "__main__":
    unittest.main()
