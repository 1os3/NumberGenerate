# NumberGenerate

NumberGenerate 是一个面向教育的 MNIST 手写数字生成项目。它用最小但完整的工程结构展示一条生成模型路线：

```text
MNIST 图像 -> 无条件 VAE 压缩到潜空间 -> 数字条件 Flow Matching 在潜空间生成 -> VAE Decoder 还原图像
```

项目刻意把配置、数据、模型、训练、采样、可视化拆开，便于沿着代码理解每个概念。所有可调参数集中在 `config/default.yaml`，所有运行产物默认写入项目目录内。

## 学习路线

建议按下面顺序阅读：

1. `config/default.yaml`：先看模型通道数、训练轮数、采样步数等全局设定。
2. `data/mnist.py`：理解 MNIST 如何被加载为 `[0,1]` 图像张量。
3. `model/layers.py`：理解 `LayerNorm2d`、`AdaLN2d`、普通残差块和条件 ConvNeXt 块。
4. `model/vae.py`：理解图像如何被压缩为潜变量，再被解码回图像。
5. `model/flow.py`：理解时间条件和数字条件如何注入 Flow 主干。
6. `train/vae_trainer.py` 与 `train/flow_trainer.py`：理解两个阶段的训练目标。
7. `train/sampling.py` 与 `vis/plots.py`：理解生成过程和可视化结果。

## 总体架构

```text
训练 VAE:
    x: [N,1,28,28]
      -> VAE Encoder
      -> mu, logvar: [N,64,14,14]
      -> reparameterize
      -> z: [N,64,14,14]
      -> VAE Decoder
      -> recon: [N,1,28,28]

训练 Flow:
    x: [N,1,28,28]
      -> frozen VAE Encoder
      -> z_1 = mu: [N,64,14,14]
    z_0 ~ N(0,I): [N,64,14,14]
    t in {0/16, 1/16, ..., 15/16}
    z_t = (1-t) * z_0 + t * z_1
      -> FlowModel(z_t, t, label)
      -> pred_velocity: [N,64,14,14]
    target_velocity = z_1 - z_0

生成采样:
    z_0 ~ N(0,I)
      -> Euler 更新 16 步
      -> z_1_hat
      -> VAE Decoder
      -> image: [N,1,28,28]
```

核心思想是：VAE 负责把图像空间变成更适合建模的潜空间；Flow 只学习潜空间里的运动方向，因此训练和采样都比直接在像素空间建模更轻量。

## 数据层

MNIST 数据由 `torchvision.datasets.MNIST` 读取，默认下载到 `datasets/`。图像只经过 `transforms.ToTensor()`，因此像素范围是 `[0,1]`，没有额外标准化。

数据张量形状：

| 名称 | 形状 | 含义 |
| --- | --- | --- |
| `images` | `[N,1,28,28]` | MNIST 灰度图 batch |
| `labels` | `[N]` | 数字标签，范围 `[0,10)` |

入口函数：

```python
from config import load_config
from data import get_mnist_loaders

cfg = load_config()
train_loader, test_loader = get_mnist_loaders(cfg)
```

## 基础层设计

基础层位于 `model/layers.py`。

### LayerNorm2d

`LayerNorm2d` 对 channels-first 特征图 `[N,C,H,W]` 做逐像素通道归一化。也就是对每个样本、每个空间位置的 `C` 个通道计算均值和方差。

```text
x: [N,C,H,W]
mean/var over C
out: [N,C,H,W]
```

这样做的教育意义是：它保留卷积网络常见的 `[N,C,H,W]` 数据格式，同时展示 LayerNorm 与卷积特征图结合的方式。

### AdaLN2d

`AdaLN2d` 是条件注入层。它先做不带可学习仿射参数的 `LayerNorm2d`，再用条件向量生成缩放和平移：

```text
cond: [N,D]
Linear(cond) -> scale, shift: [N,C]
AdaLN(x, cond) = LayerNorm2d(x) * (1 + scale) + shift
```

本项目中，AdaLN 只用于 Flow 主干块，不用于 VAE，也不用于 Flow 的输入/输出投影。这样可以清楚地区分：

- 普通卷积负责局部特征变换。
- AdaLN 负责把“当前时间步”和“要生成哪个数字”告诉网络。

### ResidualBlock

`ResidualBlock` 用在 VAE 中，是普通瓶颈残差卷积块：

```text
输入 x: [N,C,H,W]
  -> LayerNorm2d(C)
  -> 1x1 Conv(C -> C/2)
  -> 3x3 Conv(C/2 -> C/2)
  -> GELU
  -> 1x1 Conv(C/2 -> C)
  -> + x
输出: [N,C,H,W]
```

它展示了残差连接的基本作用：让卷积块学习“对输入的修正”，而不是从零构造整张特征图。

### ConditionalDepthwiseSeparableBlock

Flow 主干使用 8 个条件 ConvNeXt 风格块：

```text
输入 x: [N,C,H,W], cond: [N,D]
  -> 7x7 DepthwiseConv(C -> C)
  -> AdaLN2d(C, D)
  -> 1x1 Conv(C -> 2C)
  -> GELU
  -> 1x1 Conv(2C -> C)
  -> + x
输出: [N,C,H,W]
```

其中 `DepthwiseConv` 负责空间混合，`1x1 Conv` 负责通道混合，`AdaLN2d` 负责条件控制。

## VAE 架构

VAE 位于 `model/vae.py`。它是无条件模型，不接收数字标签，也不接收时间步。

### Encoder

| 步骤 | 层 | 输入形状 | 输出形状 | 作用 |
| --- | --- | --- | --- | --- |
| 1 | `1x1 Conv` | `[N,1,28,28]` | `[N,64,28,28]` | 提升通道数 |
| 2 | `ResidualBlock` | `[N,64,28,28]` | `[N,64,28,28]` | 提取局部结构 |
| 3 | `2x2 stride=2 Conv` | `[N,64,28,28]` | `[N,64,14,14]` | 下采样到潜空间 |
| 4 | `1x1 Conv` | `[N,64,14,14]` | `[N,64,14,14]` | 输出 `mu` |
| 5 | `1x1 Conv` | `[N,64,14,14]` | `[N,64,14,14]` | 输出 `logvar` |

`mu` 表示潜变量分布的均值，`logvar` 表示对数方差。训练时通过重参数化采样：

```text
std = exp(0.5 * logvar)
z = mu + eps * std, eps ~ N(0,I)
```

### Decoder

| 步骤 | 层 | 输入形状 | 输出形状 | 作用 |
| --- | --- | --- | --- | --- |
| 1 | `ResidualBlock` | `[N,64,14,14]` | `[N,64,14,14]` | 处理潜变量特征 |
| 2 | `1x1 Conv(64->256)` | `[N,64,14,14]` | `[N,256,14,14]` | 为像素洗牌准备通道 |
| 3 | `PixelShuffle(2)` | `[N,256,14,14]` | `[N,64,28,28]` | 上采样 2 倍 |
| 4 | `3x3 Conv` + `sigmoid` | `[N,64,28,28]` | `[N,1,28,28]` | 输出 `[0,1]` 图像 |

### VAE 损失

VAE 损失由重建误差和 KL 正则组成：

```text
recon_loss = MSE(recon, x)
kl_loss = -0.5 * mean(1 + logvar - mu^2 - exp(logvar))
loss = recon_loss + beta * kl_loss
```

默认 `beta = train.vae_beta = 0.0001`。较小的 beta 让模型优先学会清晰重建，适合教育项目先观察“编码-解码”是否工作。

## Flow Matching 架构

Flow 位于 `model/flow.py`。它不直接生成像素，而是在 VAE 潜空间 `[N,64,14,14]` 里学习速度场。

### 条件构造

Flow 接收两个条件：

- 时间 `t`：表示当前生成进度。
- 数字标签 `label`：表示要生成哪个数字。

时间嵌入使用 16 个频率，每个频率产生 sin 和 cos 两个值：

```text
gamma(t) = [sin(w_1 t), cos(w_1 t), ..., sin(w_16 t), cos(w_16 t)]
w_i = 100 ^ (-2i / 32)
```

实现中：

```text
time_embedding(t): [N,32]
time_mlp: [N,32] -> [N,128]
label_embedding(label): [N] -> [N,128]
cond = time_cond + label_cond: [N,128]
```

这个 `cond` 会传给每一个 Flow 主干块的 `AdaLN2d`。

### Flow 主体

| 步骤 | 层 | 输入形状 | 输出形状 | 是否注入条件 |
| --- | --- | --- | --- | --- |
| 1 | `1x1 Conv(64->128)` | `[N,64,14,14]` | `[N,128,14,14]` | 否 |
| 2 | 8 个条件 ConvNeXt 块 | `[N,128,14,14]` | `[N,128,14,14]` | 是 |
| 3 | `1x1 Conv(128->64)` | `[N,128,14,14]` | `[N,64,14,14]` | 否 |

只在主干块注入条件，是为了让结构更容易理解：输入投影只负责换通道，输出投影只负责回到潜空间维度，真正的条件生成能力集中在 8 个主干块。

### Flow 训练目标

训练时冻结 VAE，只用 Encoder 产生数据潜变量：

```text
z_1 = VAE.encode(x).mu
z_0 ~ N(0,I)
t = k / 16, k in {0,1,...,15}
z_t = (1 - t) * z_0 + t * z_1
target_velocity = z_1 - z_0
pred_velocity = FlowModel(z_t, t, label)
loss = MSE(pred_velocity, target_velocity)
```

这是一条线性路径：从纯噪声 `z_0` 走到真实数据潜变量 `z_1`。Flow 学的是“在任意中间位置应该往哪里走”。

### 采样过程

采样从高斯噪声开始，然后用 Euler 方法更新 16 步：

```text
z = N(0,I)
for step in 0..15:
    t = step / 16
    velocity = FlowModel(z, t, label)
    z = z + velocity / 16
image = VAE.decode(z)
```

`sample_flow(..., return_history=True)` 会保存中间图像，方便观察数字从噪声逐步成形。

## 可视化解释

可视化函数位于 `vis/plots.py`。

| 输出文件 | 函数 | 解释 |
| --- | --- | --- |
| `outputs/generation_steps.png` | `save_generation_steps` | 展示每个数字在采样 16 步中的生成轨迹 |
| `outputs/flow_feature_pca.png` | `save_flow_feature_pca` | 把 Flow 最后一层中间特征压到 2D，观察不同数字是否分离 |
| `outputs/vae_latent_pca.png` | `save_vae_latent_pca` | 把 VAE 潜变量均值压到 2D，观察 VAE 是否形成可分结构 |

PCA 使用 `torch.pca_lowrank`。散点颜色对应 MNIST 标签。

## 目录结构

```text
config/              配置加载、类型定义和默认参数
data/                MNIST 数据集与 DataLoader
model/               LayerNorm2d、AdaLN2d、VAE、FlowModel
train/               训练入口、checkpoint 工具、采样函数
vis/                 生成过程和 PCA 可视化
tests/               配置、形状、损失和条件注入测试
Doc/                 开发规范与项目索引
datasets/            MNIST 下载目录，运行后生成，默认被 git 忽略
ckpt/                训练权重目录，运行后生成，默认被 git 忽略
outputs/             可视化输出目录，运行后生成，默认被 git 忽略
logs/                日志目录，运行后生成，默认被 git 忽略
```

## 关键配置

默认配置位于 `config/default.yaml`。所有实验参数都应只在配置文件中修改。

| 配置键 | 默认值 | 说明 |
| --- | --- | --- |
| `paths.data_dir` | `datasets` | MNIST 下载目录 |
| `paths.checkpoint_dir` | `ckpt` | 权重保存目录 |
| `paths.output_dir` | `outputs` | 可视化输出目录 |
| `data.batch_size` | `128` | 训练 batch size |
| `train.device` | `auto` | 自动选择 CUDA 或 CPU |
| `train.vae_epochs` | `30` | VAE 训练轮数 |
| `train.flow_epochs` | `50` | Flow 训练轮数 |
| `train.vae_lr` | `0.001` | VAE 学习率 |
| `train.flow_lr` | `0.0002` | Flow 学习率 |
| `train.max_train_steps` | `null` | 调试时限制总训练步数 |
| `model.latent_channels` | `64` | VAE 潜变量通道数 |
| `model.flow_hidden_channels` | `128` | Flow 主干通道数 |
| `model.flow_depth` | `8` | Flow 条件主干块数量 |
| `model.time_frequencies` | `16` | 时间嵌入频率数 |
| `model.condition_dim` | `128` | AdaLN 条件向量维度 |
| `sample.sampling_steps` | `16` | 训练和采样使用的离散时间步 |

调试配置示例 `config/debug.yaml`：

```yaml
data:
  batch_size: 16
  num_workers: 0
train:
  device: cpu
  vae_epochs: 1
  flow_epochs: 1
  max_train_steps: 5
```

命令行通过 `--config` 使用覆盖配置。

## 安装部署

### 环境要求

- Python 3.12 已验证。
- CPU 可以跑通测试和小规模训练。
- 完整训练建议使用 CUDA。

安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如果已有 `.venv`：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 新机器部署流程

1. 克隆或复制项目到目标目录。
2. 进入项目根目录。
3. 创建并激活虚拟环境，或使用已有 Python 环境。
4. 执行 `pip install -r requirements.txt`。
5. 运行 `python -m unittest discover -s tests` 确认环境可用。
6. 运行 `python -m train.vae_trainer` 生成 `ckpt/vae.pt`。
7. 运行 `python -m train.flow_trainer` 生成 `ckpt/flow.pt`。
8. 调用可视化 API，检查 `outputs/` 下的生成图和 PCA 图。

项目所有运行产物默认都写入项目目录内。`datasets/`、`ckpt/`、`outputs/`、`logs/` 已被 `.gitignore` 忽略。

## 训练命令

训练 VAE：

```powershell
.\.venv\Scripts\python.exe -m train.vae_trainer
```

训练完成后生成：

```text
ckpt/vae.pt
```

训练 Flow：

```powershell
.\.venv\Scripts\python.exe -m train.flow_trainer
```

训练完成后生成：

```text
ckpt/flow.pt
```

使用调试配置：

```powershell
.\.venv\Scripts\python.exe -m train.vae_trainer --config config/debug.yaml
.\.venv\Scripts\python.exe -m train.flow_trainer --config config/debug.yaml
```

## 生成与可视化命令

先确保 `ckpt/vae.pt` 和 `ckpt/flow.pt` 已存在。

```powershell
.\.venv\Scripts\python.exe -m vis.visualize
```

使用调试配置或自定义配置：

```powershell
.\.venv\Scripts\python.exe -m vis.visualize --config config/debug.yaml
```

默认输出：

```text
outputs/generation_steps.png
outputs/flow_feature_pca.png
outputs/vae_latent_pca.png
```

如果缺少 checkpoint，命令会提示先运行：

```powershell
.\.venv\Scripts\python.exe -m train.vae_trainer
.\.venv\Scripts\python.exe -m train.flow_trainer
```

直接在代码中调用采样函数：

```python
import torch

from config import load_config
from model import FlowModel, VAE
from train import sample_flow
from train.common import load_checkpoint, prepare_runtime

cfg = load_config()
device = prepare_runtime(cfg)
vae = VAE(cfg).to(device)
flow = FlowModel(cfg).to(device)
vae.load_state_dict(load_checkpoint(cfg.paths.vae_checkpoint, device)["model"])
flow.load_state_dict(load_checkpoint(cfg.paths.flow_checkpoint, device)["model"])

labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
images = sample_flow(flow, vae, labels, cfg)
```

## 测试

运行所有测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

运行语法编译检查：

```powershell
.\.venv\Scripts\python.exe -m compileall -q config data model train vis tests
```

当前测试覆盖：

- 默认配置可加载。
- VAE encode/decode/forward 形状正确。
- Flow 主干块全部是条件块，输出速度形状正确。
- VAE loss 和 Flow Matching loss 可反向传播。
- `sample_flow` 输出 `N x 1 x 28 x 28` 图像。

## 开发规范

开发时遵守 `Doc/开发规范.md`：

- 配置值只写在 `config/default.yaml` 或覆盖配置中。
- 有实际逻辑的 `.py` 文件必须有中文文件头。
- 实现文件的校验逻辑放入同目录同名前缀的 `_checks.py`。
- 新增或删除源文件时同步更新 `Doc/Index.md`。
- 普通归一化使用 `LayerNorm2d`，条件注入使用 `AdaLN2d`。
