# ByteDrive 文档索引

全项目文档与源文件的单一导航入口。**每次增删文件，同一提交内更新本表。**
每行格式：`相对路径 — 一句话职责`（职责应与该文件文件头首行一致）。
校验伴随文件 `X_checks.py` 附属于 `X.py`，**不单独列入索引**。

## 规范与文档

- [README.md](../README.md) — NumberGenerate 项目说明、教育性架构讲解与部署流程
- [Doc/开发规范.md](开发规范.md) — 项目强制开发规范（文档/注释/配置/校验/简洁）
- [Doc/Index.md](Index.md) — 本文档索引

## config/ — 配置与校验（参数唯一来源）

- [config/default.yaml](../config/default.yaml) — 全项目默认配置的唯一数据来源
- [config/schema.py](../config/schema.py) — 定义项目配置结构并在加载期完成参数校验
- [config/loader.py](../config/loader.py) — 加载 YAML 配置并返回强类型配置对象

## data/ — 数据读取与预处理

- [data/mnist.py](../data/mnist.py) — 构建二通道有/无表示的 MNIST 数据集与 DataLoader

## model/ — 网络结构定义

- [model/layers.py](../model/layers.py) — 提供 VAE 与 Flow 共用的卷积层和归一化层
- [model/vae.py](../model/vae.py) — 定义输入输出为 background/foreground 二通道概率图的无条件卷积 VAE
- [model/flow.py](../model/flow.py) — 定义数字条件潜空间 Flow Matching 模型

## train/ — 训练 / 评估循环

- [train/common.py](../train/common.py) — 提供训练脚本共用的随机种子、设备、checkpoint 保存、断点续训和梯度诊断工具
- [train/vae_trainer.py](../train/vae_trainer.py) — 训练无条件 MNIST VAE
- [train/flow_trainer.py](../train/flow_trainer.py) — 训练数字条件潜空间 Flow Matching 模型
- [train/sampling.py](../train/sampling.py) — 提供 Flow Matching 生成采样函数

## vis/ — 可视化与日志渲染

- [vis/plots.py](../vis/plots.py) — 保存生成结果、Flow 特征 PCA 图、VAE 重构、VAE PCA 和潜变量空间诊断可视化
- [vis/visualize.py](../vis/visualize.py) — 一键加载训练权重并按模式生成完整或仅 VAE 可视化图

## tests/ — 自动化验证

- [tests/test_config_and_shapes.py](../tests/test_config_and_shapes.py) — 验证默认配置、模型形状、条件注入和核心损失
