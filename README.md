# NumberGenerate

NumberGenerate 是一个面向教育的 MNIST 手写数字生成项目。它用最小但完整的工程结构展示一条生成模型路线：

```text
MNIST 图像 -> 无条件 VAE 压缩到潜空间 -> 数字条件 Flow Matching 在潜空间生成 -> VAE Decoder 还原图像
```

项目刻意把配置、数据、模型、训练、采样、可视化拆开，便于沿着代码理解每个概念。所有可调参数集中在 `config/default.yaml`，所有运行产物默认写入项目目录内。

## 学习路线

建议按下面顺序阅读：

1. `config/default.yaml`：先看模型通道数、训练轮数、采样步数等全局设定。
2. `data/mnist.py`：理解 MNIST 如何被二值化为 `[background, foreground]` 二通道张量。
3. `model/layers.py`：理解 `LayerNorm2d`、`AdaLN2d`、普通残差块和条件 ConvNeXt 块。
4. `model/vae.py`：理解图像如何被压缩为潜变量，再被解码回图像。
5. `model/flow.py`：理解时间条件和数字条件如何注入 Flow 主干。
6. `train/vae_trainer.py` 与 `train/flow_trainer.py`：理解两个阶段的训练目标。
7. `train/sampling.py` 与 `vis/plots.py`：理解生成过程和可视化结果。

## 总体架构

```text
训练 VAE:
    x: [N,2,28,28]
      -> VAE Encoder
      -> Encoder LayerNorm2d
      -> mu, logvar: [N,32,14,14]
      -> reparameterize
      -> z: [N,32,14,14]
      -> VAE Decoder
      -> recon: [N,2,28,28]

训练 Flow:
    x: [N,2,28,28]
      -> frozen VAE Encoder
      -> mu, logvar: [N,32,14,14]
      -> z_1 = mu + eps_post * exp(0.5 * logvar)
    z_0 ~ N(0,I): [N,32,14,14]
    t ~ Uniform(0,1)
    z_t = (1-t) * z_0 + t * z_1
      -> FlowModel(z_t, t, label)
      -> pred_velocity: [N,32,14,14]
    target_velocity = z_1 - z_0

生成采样:
    z_0 ~ N(0,I)
      -> Euler 更新 sample.sampling_steps 步
      -> z_1_hat
      -> VAE Decoder
      -> image: [N,2,28,28]
```

核心思想是：VAE 负责把图像空间变成更适合建模的潜空间；Flow 只学习潜空间里的运动方向，因此训练和采样都比直接在像素空间建模更轻量。

## 数据层

MNIST 数据由 `torchvision.datasets.MNIST` 读取，默认下载到 `datasets/`。原始灰度图先经过 `transforms.ToTensor()` 得到 `[0,1]`，再按 `data.binarize_threshold` 二值化成 `[background, foreground]` 二通道 one-hot 张量。背景不再是“没有数值”，而是第 0 通道上的明确类别。

数据张量形状：

| 名称 | 形状 | 含义 |
| --- | --- | --- |
| `images` | `[N,2,28,28]` | MNIST 二通道有/无 batch，第 0 通道是 background，第 1 通道是 foreground |
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
| 1 | `1x1 Conv` | `[N,2,28,28]` | `[N,64,28,28]` | 提升通道数 |
| 2 | `ResidualBlock` | `[N,64,28,28]` | `[N,64,28,28]` | 提取局部结构 |
| 3 | `2x2 stride=2 Conv` | `[N,64,28,28]` | `[N,32,14,14]` | 下采样到潜空间 |
| 4 | `LayerNorm2d` | `[N,32,14,14]` | `[N,32,14,14]` | 稳定潜空间特征尺度 |
| 5 | `1x1 Conv` | `[N,32,14,14]` | `[N,32,14,14]` | 输出 `mu` |
| 6 | `1x1 Conv` | `[N,32,14,14]` | `[N,32,14,14]` | 输出 `logvar` |

Encoder 在输出 `mu/logvar` 前先对潜空间特征做一次 `LayerNorm2d`，用于抑制下采样后特征尺度漂移。`mu` 表示潜变量分布的均值，`logvar` 表示对数方差。训练时通过重参数化采样：

```text
std = exp(0.5 * logvar)
z = mu + eps * std, eps ~ N(0,I)
```

### Decoder

| 步骤 | 层 | 输入形状 | 输出形状 | 作用 |
| --- | --- | --- | --- | --- |
| 1 | `ResidualBlock` | `[N,32,14,14]` | `[N,32,14,14]` | 处理潜变量特征 |
| 2 | `1x1 Conv(32->256)` | `[N,32,14,14]` | `[N,256,14,14]` | 为像素洗牌准备通道 |
| 3 | `PixelShuffle(2)` | `[N,256,14,14]` | `[N,64,28,28]` | 上采样 2 倍 |
| 4 | `3x3 Conv` + `sigmoid` | `[N,64,28,28]` | `[N,2,28,28]` | 输出 background/foreground 概率图 |

### VAE 损失

VAE 损失由重建误差和 KL 正则组成：

```text
recon_loss = BCE(recon, x)
kl_loss = -0.5 * mean(1 + logvar - mu^2 - exp(logvar))
loss = recon_loss + kl_weight * kl_loss
```

默认 `kl_weight = train.vae_kl_weight = 0.001`。较小的 KL 权重让模型优先学会清晰重建，适合教育项目先观察“编码-解码”是否工作。

## Flow Matching 架构

Flow 位于 `model/flow.py`。它不直接生成像素，而是在 VAE 潜空间 `[N,32,14,14]` 里学习速度场。

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
| 1 | `1x1 Conv(32->256)` | `[N,32,14,14]` | `[N,256,14,14]` | 否 |
| 2 | 8 个条件 ConvNeXt 块 | `[N,256,14,14]` | `[N,256,14,14]` | 是 |
| 3 | `LayerNorm2d` | `[N,256,14,14]` | `[N,256,14,14]` | 否 |
| 4 | `1x1 Conv(256->32)` | `[N,256,14,14]` | `[N,32,14,14]` | 否 |

只在主干块注入条件，是为了让结构更容易理解：输入投影只负责换通道，输出端 `LayerNorm2d` 负责压住主干特征尺度，输出投影只负责回到潜空间维度，真正的条件生成能力集中在 8 个主干块。

### Flow 训练目标

训练时冻结 VAE，用 Encoder 输出的 `mu/logvar` 采样 posterior 潜变量作为数据端：

```text
mu, logvar = VAE.encode(x)
eps_post ~ N(0,I)
z_1 = mu + eps_post * exp(0.5 * logvar)
z_0 ~ N(0,I)
t ~ Uniform(0,1)
z_t = (1 - t) * z_0 + t * z_1
target_velocity = z_1 - z_0
pred_velocity = FlowModel(z_t, t, label)
loss = MSE(pred_velocity, target_velocity)
```

这是一条线性路径：从纯噪声 `z_0` 走到 VAE posterior 样本 `z_1`。训练时的 `t` 是连续随机时间，不由 `sample.sampling_steps` 离散化；`sample.sampling_steps` 只控制采样推理时 Euler 更新多少步。Flow 学的是“在任意中间位置应该往哪里走”。

这里有两类噪声：

- `eps_post` 是 VAE posterior 采样噪声，形状与 `mu/logvar` 相同。它不是直接加到图像上，而是先乘以 `std = exp(0.5 * logvar)`，再加到 `mu` 上得到数据端潜变量 `z_1`。
- `z_0` 是 Flow 起点噪声，直接从标准高斯采样，形状也是 `[N,32,14,14]`。Flow 训练的线性路径从这个噪声端走向 posterior 样本端。

因此训练时没有给 MNIST 像素图直接加噪声；噪声全部发生在 VAE 潜空间里。

### 采样过程

采样从高斯噪声开始，然后用 Euler 方法更新 `sample.sampling_steps` 步：

```text
z = N(0,I)
for step in 0..sample.sampling_steps-1:
    t = step / sample.sampling_steps
    velocity = FlowModel(z, t, label)
    z = z + velocity / sample.sampling_steps
image = VAE.decode(z)
```

`sample_flow(..., return_history=True)` 会按 `sample.history_steps` 均匀保存中间图像，方便观察数字从噪声逐步成形。默认配置让 `history_steps` 与 `sampling_steps` 一致；如果只想抽帧展示，可以把 `history_steps` 设得更小。

`sample_flow_step_trace(...)` 则始终记录全部 `sample.sampling_steps` 的预测速度和最后一个 Flow 主干块输出，供逐步诊断可视化使用，不受 `sample.history_steps` 抽帧设置影响。

## 可视化解释

可视化函数位于 `vis/plots.py`。

| 输出文件 | 函数 | 解释 |
| --- | --- | --- |
| `outputs/generation_steps.png` | `save_generation_steps` | 展示每个数字在 `sample.history_steps` 个历史帧中的生成轨迹 |
| `outputs/flow_prediction_steps.png` | `save_flow_prediction_steps` | 展示指定数字在每个采样步的预测流场和末端骨干特征；流场以共享 PCA 投影为二维箭头，特征以共享 PCA 投影为 RGB |
| `outputs/flow_feature_pca_map.png` | `save_flow_feature_pca_map` | 把单个 posterior 样本的 Flow 末端特征图用 PCA 压到 3 通道 RGB |
| `outputs/flow_feature_pca.png` | `save_flow_feature_pca` | 把 posterior 样本对应的 Flow 最后一层中间特征压到 2D，观察不同数字是否分离 |
| `outputs/vae_reconstruction.png` | `save_vae_reconstruction` | 选一张 MNIST，展示原图、VAE 压缩表示、重构图和误差图 |
| `outputs/vae_latent_pca.png` | `save_vae_latent_pca` | 把 VAE 潜变量均值压到 2D，观察 VAE 是否形成可分结构 |
| `outputs/vae_latent_distribution.png` | `save_vae_latent_distribution` | 按 MNIST 前景/背景分别展示 VAE 潜变量均值、posterior 标准差和 KL 分布 |
| `outputs/vae_kl_map.png` | `save_vae_kl_map` | 把每个潜空间位置的 KL 强度画成热力图，观察信息是否集中在笔画区域 |
| `outputs/vae_latent_energy_map.png` | `save_vae_latent_energy_map` | 把 `mean(abs(mu))` 画成空间热力图，观察潜变量能量是否跟前景结构对齐 |

PCA 使用 `torch.pca_lowrank`。散点颜色对应 MNIST 标签。

逐步预测可视化跟踪 `visual.flow_step_label` 指定的数字。所有采样步共用 PCA 基底、流场幅值范围和 RGB 色阶，因此相邻步骤之间的箭头、亮度与颜色可以直接比较。每个时间步上方是潜空间预测速度投影后的二维流场，下方是该次预测最后一个主干块的输出特征。

Flow 特征图可视化会从测试集取一个样本，在 `visual.feature_map_time` 指定的时间点提取 Flow 末端特征，用 PCA 压成 3 通道图像。这样比直接画任意几个卷积通道更容易观察整体结构。

VAE 重构可视化会从 MNIST 测试集中取一张图，展示 `foreground 原图 -> 32x14x14 潜变量 PCA 图 -> foreground 重构图 -> 绝对误差图`，用于观察压缩和重构能力。可视化默认展示 foreground 通道，因为 background 通道表示“无笔画”的概率。

VAE 分布可视化不会把所有潜变量位置粗暴混在一起统计，而是先把 MNIST 图像用 area 下采样到 `14x14`，再用阈值区分前景笔画和背景区域。这样可以避免背景像素数量过多导致直方图被 0 附近的背景潜变量主导。

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
| `data.binarize_threshold` | `0.5` | MNIST 二值化阈值，大于该值记为 foreground |
| `train.device` | `auto` | 自动选择 CUDA 或 CPU |
| `train.vae_epochs` | `30` | VAE 训练轮数 |
| `train.flow_epochs` | `50` | Flow 训练轮数 |
| `train.vae_lr` | `0.0001` | VAE 学习率 |
| `train.flow_lr` | `0.0001` | Flow 学习率 |
| `train.vae_kl_weight` | `0.001` | VAE KL 散度损失权重 |
| `train.grad_monitor_enabled` | `true` | 是否在训练日志中监测小梯度比例 |
| `train.grad_small_threshold` | `0.00000001` | 小梯度判定阈值，`abs(grad)` 小于该值会计入 small ratio |
| `train.grad_small_warn_ratio` | `0.99` | 小梯度比例超过该值时打印提示 |
| `train.max_train_steps` | `null` | 调试时限制总训练步数 |
| `model.image_channels` | `2` | VAE 输入/输出通道数，对应 `[background, foreground]` |
| `model.latent_channels` | `32` | VAE 潜变量通道数，也是 Flow 输入/输出投影层的端点通道数 |
| `model.flow_hidden_channels` | `256` | Flow 主干通道数；输入投影 `32->256`，输出投影 `256->32` |
| `model.flow_depth` | `8` | Flow 条件主干块数量 |
| `model.time_frequencies` | `16` | 时间嵌入频率数 |
| `model.condition_dim` | `128` | AdaLN 条件向量维度 |
| `sample.sampling_steps` | `32` | 采样推理时的 Euler 更新步数；Flow 训练时间 `t` 连续随机采样 |
| `sample.history_steps` | `32` | 生成过程可视化保存的历史帧数，通常与 `sample.sampling_steps` 一致 |
| `visual.feature_map_channels` | `3` | Flow 末端特征图 PCA 保留通道数，最终取 3 通道组成 RGB 图 |
| `visual.feature_map_time` | `0.5` | 特征图可视化采用的生成时间点 |
| `visual.flow_step_label` | `0` | 逐步流场与末端骨干特征可视化跟踪的数字标签 |

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
6. 运行 `python -m train.vae_trainer` 生成并持续更新 `ckpt/vae.pt`。
7. 运行 `python -m train.flow_trainer` 生成并持续更新 `ckpt/flow.pt`。
8. 运行 `python -m vis.visualize` 检查完整生成图和 PCA 图；只检查 VAE 时运行 `python -m vis.visualize --mode vae`。

项目所有运行产物默认都写入项目目录内。`datasets/`、`ckpt/`、`outputs/`、`logs/` 已被 `.gitignore` 忽略。

## 训练命令

训练 VAE：

```powershell
.\.venv\Scripts\python.exe -m train.vae_trainer
```

每个 epoch 结束后都会覆盖保存：

```text
ckpt/vae.pt
```

训练 Flow：

```powershell
.\.venv\Scripts\python.exe -m train.flow_trainer
```

每个 epoch 结束后都会覆盖保存：

```text
ckpt/flow.pt
```

使用调试配置：

```powershell
.\.venv\Scripts\python.exe -m train.vae_trainer --config config/debug.yaml
.\.venv\Scripts\python.exe -m train.flow_trainer --config config/debug.yaml
```

两个训练入口都会在启动时自动查找对应的最新 checkpoint。若 `ckpt/vae.pt` 或 `ckpt/flow.pt` 已存在，程序会恢复模型参数、优化器状态和 `epoch/step/loss` 指标，并从下一轮 epoch 继续训练。checkpoint 使用单文件覆盖策略，因此默认只保留最新训练状态；如果想从头训练，删除或重命名对应的 checkpoint 文件即可。

训练日志会在 `train.log_interval` 指定的步数上输出梯度诊断字段。`grad_mean_abs` 和 `grad_max_abs` 描述梯度幅度，`grad_small_ratio` 表示低于 `train.grad_small_threshold` 的梯度元素比例，`grad_zero_ratio` 表示严格为 0 的比例。监测发生在 `loss.backward()` 之后、梯度裁剪之前，因此它反映的是未裁剪的原始梯度。

## 生成与可视化命令

完整可视化需要 `ckpt/vae.pt` 和 `ckpt/flow.pt` 都已存在。

```powershell
.\.venv\Scripts\python.exe -m vis.visualize
```

仅可视化 VAE 时只需要 `ckpt/vae.pt`：

```powershell
.\.venv\Scripts\python.exe -m vis.visualize --mode vae
.\.venv\Scripts\python.exe -m vis.visualize --only-vae
```

使用调试配置或自定义配置：

```powershell
.\.venv\Scripts\python.exe -m vis.visualize --config config/debug.yaml
.\.venv\Scripts\python.exe -m vis.visualize --config config/debug.yaml --mode vae
```

默认输出：

```text
outputs/generation_steps.png
outputs/flow_prediction_steps.png
outputs/flow_feature_pca_map.png
outputs/flow_feature_pca.png
outputs/vae_reconstruction.png
outputs/vae_latent_pca.png
outputs/vae_latent_distribution.png
outputs/vae_kl_map.png
outputs/vae_latent_energy_map.png
```

VAE-only 模式只输出：

```text
outputs/vae_reconstruction.png
outputs/vae_latent_pca.png
outputs/vae_latent_distribution.png
outputs/vae_kl_map.png
outputs/vae_latent_energy_map.png
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
- Flow 逐步轨迹完整记录每个采样步的预测速度和末端骨干特征。
- VAE loss 和 Flow Matching loss 可反向传播。
- `sample_flow` 输出 `N x 2 x 28 x 28` 的 background/foreground 概率图。

## 开发规范

开发时遵守 `Doc/开发规范.md`：

- 配置值只写在 `config/default.yaml` 或覆盖配置中。
- 有实际逻辑的 `.py` 文件必须有中文文件头。
- 实现文件的校验逻辑放入同目录同名前缀的 `_checks.py`。
- 新增或删除源文件时同步更新 `Doc/Index.md`。
- 普通归一化使用 `LayerNorm2d`，条件注入使用 `AdaLN2d`。
