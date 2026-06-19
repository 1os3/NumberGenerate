"""一键加载训练权重并生成全部可视化图。

模块: vis/visualize.py
依赖: argparse, pathlib, config, data, model, train.common, vis.plots, vis.visualize_checks
读取配置: train.seed, train.device, paths.checkpoint_dir, paths.output_dir, paths.log_dir, paths.vae_checkpoint, paths.flow_checkpoint, data.batch_size, data.num_workers, data.download, data.pin_memory, visual.feature_map_channels, visual.feature_map_time, visual.pca_samples
对外接口:
    - run_visualization(cfg, mode="all") -> dict[str, Path]
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
from vis.plots import (
    save_flow_feature_pca_map,
    save_flow_feature_pca,
    save_generation_steps,
    save_vae_latent_distribution,
    save_vae_latent_pca,
    save_vae_reconstruction,
)
from vis.visualize_checks import check_visualization_checkpoints


def run_visualization(cfg: AppConfig, mode: str = "all") -> dict[str, Path]:
    """加载 VAE/Flow 权重并生成全部可视化文件。

    参数:
        cfg: 项目配置对象，读取设备、checkpoint 和可视化输出路径
    返回:
        可视化名称到输出路径的映射
    """

    check_visualization_checkpoints(cfg, mode)
    device = prepare_runtime(cfg)
    vae = VAE(cfg).to(device)
    vae.load_state_dict(load_checkpoint(cfg.paths.vae_checkpoint, device)["model"])
    vae.eval()
    _, test_loader = get_mnist_loaders(cfg)
    vae_outputs = {
        "vae_reconstruction": save_vae_reconstruction(vae, test_loader, cfg),
        "vae_latent_pca": save_vae_latent_pca(vae, test_loader, cfg),
        "vae_latent_distribution": save_vae_latent_distribution(vae, test_loader, cfg),
    }
    if mode == "vae":
        return vae_outputs
    flow = FlowModel(cfg).to(device)
    flow.load_state_dict(load_checkpoint(cfg.paths.flow_checkpoint, device)["model"])
    flow.eval()
    outputs = {
        "generation_steps": save_generation_steps(flow, vae, cfg),
        "flow_feature_pca_map": save_flow_feature_pca_map(flow, vae, test_loader, cfg),
        "flow_feature_pca": save_flow_feature_pca(flow, vae, test_loader, cfg),
    }
    outputs.update(vae_outputs)
    return outputs


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser(description="加载训练权重并生成 MNIST 可视化结果")
    parser.add_argument("--config", type=str, default=None, help="可选 YAML 覆盖配置")
    parser.add_argument(
        "--mode",
        choices=["all", "vae"],
        default="all",
        help="可视化模式: all 需要 VAE 和 Flow 权重；vae 只需要 VAE 权重",
    )
    parser.add_argument(
        "--only-vae",
        action="store_true",
        help="等价于 --mode vae，只输出 VAE 重构、PCA 和分布图",
    )
    args = parser.parse_args()
    mode = "vae" if args.only_vae else args.mode
    try:
        outputs = run_visualization(load_config(args.config), mode=mode)
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
        print(f"可视化失败: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
