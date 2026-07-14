"""验证默认配置、模型形状、条件注入和核心损失。

模块: tests/test_config_and_shapes.py
依赖: unittest, torch, config, model, train
读取配置: config/default.yaml
对外接口:
    - ConfigAndShapeTests(unittest.TestCase) -> None
说明: 测试不下载 MNIST，只验证可在随机张量上跑通的核心路径。
"""

from __future__ import annotations

from dataclasses import replace
import unittest
from pathlib import Path

import torch

from config import load_config
from config.schema import build_config
from data import mnist_to_presence_channels
from model import ConditionalDepthwiseSeparableBlock, FlowModel, LayerNorm2d, VAE
from train import (
    flow_matching_loss,
    sample_flow,
    sample_flow_step_trace,
    sample_flow_time,
    sample_vae_posterior,
    vae_loss,
)
from train.common import load_latest_training_state, save_checkpoint, summarize_gradients
from vis.visualize_checks import check_visualization_checkpoints


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
        images = _binary_presence_batch(2)
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
        traced_pred, features = flow.predict_with_features(z_t, t, labels)
        target = torch.randn_like(pred)
        self.assertEqual(tuple(pred.shape), tuple(z_t.shape))
        self.assertTrue(torch.allclose(pred, traced_pred))
        self.assertEqual(len(features), self.cfg.model.flow_depth)
        self.assertEqual(tuple(features[-1].shape), (2, 256, 14, 14))
        loss = flow_matching_loss(pred, target)
        loss.backward()
        self.assertEqual(loss.ndim, 0)

    def test_sample_flow_shape(self) -> None:
        """采样函数应输出 [N,2,28,28] 二通道图像。"""

        vae = VAE(self.cfg)
        flow = FlowModel(self.cfg)
        labels = torch.tensor([0, 1], dtype=torch.long)
        images = sample_flow(flow, vae, labels, self.cfg)
        self.assertEqual(tuple(images.shape), (2, 2, 28, 28))

    def test_sample_flow_history_uses_configured_frame_count(self) -> None:
        """生成历史应精确返回 sample.history_steps 个均匀抽样帧。"""

        cfg = replace(
            self.cfg,
            sample=replace(self.cfg.sample, sampling_steps=5, history_steps=3),
        )
        vae = VAE(cfg)
        flow = FlowModel(cfg)
        labels = torch.tensor([0, 1], dtype=torch.long)
        images, history = sample_flow(flow, vae, labels, cfg, return_history=True)
        self.assertEqual(tuple(images.shape), (2, 2, 28, 28))
        self.assertEqual(tuple(history.shape), (3, 2, 2, 28, 28))

    def test_sample_flow_step_trace_records_every_step(self) -> None:
        """逐步轨迹应记录每次预测速度和最后一个主干块输出。"""

        flow = FlowModel(self.cfg)
        labels = torch.tensor([0, 1], dtype=torch.long)
        velocities, features = sample_flow_step_trace(flow, labels, self.cfg)
        self.assertEqual(tuple(velocities.shape), (2, 2, 32, 14, 14))
        self.assertEqual(tuple(features.shape), (2, 2, 256, 14, 14))

    def test_mnist_transform_creates_presence_channels(self) -> None:
        """MNIST transform 应输出 background/foreground 二通道 one-hot 表示。"""

        image = torch.tensor([[[0.0, 0.75], [0.25, 1.0]]])
        transformed = mnist_to_presence_channels(image, threshold=0.5)
        expected = torch.tensor(
            [
                [[1.0, 0.0], [1.0, 0.0]],
                [[0.0, 1.0], [0.0, 1.0]],
            ]
        )
        self.assertTrue(torch.equal(transformed, expected))

    def test_flow_training_time_is_continuous(self) -> None:
        """Flow 训练时间应从 [0,1) 连续均匀采样。"""

        times = sample_flow_time(128, torch.device("cpu"))
        self.assertEqual(tuple(times.shape), (128,))
        self.assertTrue(torch.all(times >= 0))
        self.assertTrue(torch.all(times < 1))
        scaled = times * 32
        non_grid = torch.abs(scaled - scaled.round()) > 1e-6
        self.assertGreater(int(non_grid.sum().item()), 120)

    def test_training_checkpoint_resume_restores_state(self) -> None:
        """断点续训应恢复模型参数、优化器状态和最新指标。"""

        path = self.cfg.project_root / "outputs" / "unit_test_resume.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        loss = model(torch.ones(1, 2)).sum()
        loss.backward()
        optimizer.step()
        metrics = {"epoch": 2, "step": 7, "loss": 0.5}
        try:
            save_checkpoint(
                path,
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "metrics": metrics,
                },
            )
            restored_model = torch.nn.Linear(2, 1)
            restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.01)
            restored_metrics = load_latest_training_state(
                path,
                restored_model,
                restored_optimizer,
                torch.device("cpu"),
                "Unit",
            )
            self.assertEqual(restored_metrics, metrics)
            for expected, actual in zip(model.parameters(), restored_model.parameters()):
                self.assertTrue(torch.allclose(expected, actual))
            self.assertTrue(restored_optimizer.state_dict()["state"])
        finally:
            if path.exists():
                path.unlink()

    def test_gradient_summary_detects_small_gradients(self) -> None:
        """梯度监测应能统计小梯度和零梯度比例。"""

        model = torch.nn.Linear(3, 1)
        output = model(torch.zeros(2, 3)).sum()
        output.backward()
        stats = summarize_gradients(model, small_threshold=1e-8)
        self.assertGreater(stats["grad_elements"], 0)
        self.assertGreater(stats["grad_small_ratio"], 0.0)
        self.assertGreater(stats["grad_zero_ratio"], 0.0)
        self.assertGreater(stats["grad_max_abs"], 0.0)

    def test_visualization_vae_mode_does_not_require_flow_checkpoint(self) -> None:
        """VAE-only 可视化只应检查 VAE 权重，不应要求 Flow 权重。"""

        vae_path = self.cfg.project_root / "outputs" / "unit_test_vae_only.pt"
        flow_path = self.cfg.project_root / "outputs" / "unit_test_missing_flow.pt"
        vae_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            vae_path.touch()
            if flow_path.exists():
                flow_path.unlink()
            cfg = replace(
                self.cfg,
                paths=replace(
                    self.cfg.paths,
                    vae_checkpoint=vae_path,
                    flow_checkpoint=flow_path,
                ),
            )
            check_visualization_checkpoints(cfg, "vae")
            with self.assertRaises(FileNotFoundError):
                check_visualization_checkpoints(cfg, "all")
        finally:
            if vae_path.exists():
                vae_path.unlink()


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
            "binarize_threshold": 0.5,
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
            "grad_monitor_enabled": True,
            "grad_small_threshold": 0.00000001,
            "grad_small_warn_ratio": 0.99,
            "log_interval": 1,
            "max_train_steps": 1,
        },
        "model": {
            "image_channels": 2,
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
            "flow_step_label": 0,
        },
    }
    return build_config(raw, Path.cwd())


def _binary_presence_batch(batch_size: int) -> torch.Tensor:
    images = torch.zeros(batch_size, 2, 28, 28)
    images[:, 0] = 1.0
    images[:, 0, 8:20, 10:18] = 0.0
    images[:, 1, 8:20, 10:18] = 1.0
    return images


if __name__ == "__main__":
    unittest.main()
