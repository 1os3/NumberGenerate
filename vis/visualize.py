"""一键加载训练权重并生成全部可视化图。

模块: vis/visualize.py
依赖: argparse, pathlib, config, data, model, train.common, vis.plots, vis.visualize_checks
读取配置: train.seed, train.device, paths.checkpoint_dir, paths.output_dir, paths.log_dir, paths.vae_checkpoint, paths.flow_checkpoint, data.batch_size, data.num_workers, data.download, data.pin_memory
对外接口:
    - run_visualization(cfg) -> dict[str, Path]
说明: 该入口对齐训练脚本用法，用户只需运行 python -m vis.visualize。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import AppConfig, load_config
from data import get_mnist_loaders
from model import FlowModel, VAE
from train.common import load_checkpoint, prepare_runtime
from vis.plots import save_flow_feature_pca, save_generation_steps, save_vae_latent_pca
from vis.visualize_checks import check_visualization_checkpoints


def run_visualization(cfg: AppConfig) -> dict[str, Path]:
    """加载 VAE/Flow 权重并生成全部可视化文件。

    参数:
        cfg: 项目配置对象，读取设备、checkpoint 和可视化输出路径
    返回:
        可视化名称到输出路径的映射
    """

    check_visualization_checkpoints(cfg)
    device = prepare_runtime(cfg)
    vae = VAE(cfg).to(device)
    flow = FlowModel(cfg).to(device)
    vae.load_state_dict(load_checkpoint(cfg.paths.vae_checkpoint, device)["model"])
    flow.load_state_dict(load_checkpoint(cfg.paths.flow_checkpoint, device)["model"])
    vae.eval()
    flow.eval()
    _, test_loader = get_mnist_loaders(cfg)
    return {
        "generation_steps": save_generation_steps(flow, vae, cfg),
        "flow_feature_pca": save_flow_feature_pca(flow, vae, test_loader, cfg),
        "vae_latent_pca": save_vae_latent_pca(vae, test_loader, cfg),
    }


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser(description="加载训练权重并生成 MNIST 可视化结果")
    parser.add_argument("--config", type=str, default=None, help="可选 YAML 覆盖配置")
    args = parser.parse_args()
    try:
        outputs = run_visualization(load_config(args.config))
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
        print(f"可视化失败: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
