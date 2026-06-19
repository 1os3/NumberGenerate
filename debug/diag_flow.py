"""诊断 Flow 训练是否收敛：测量潜变量统计与 Flow Matching 损失下界。

运行: python debug/diag_flow.py
"""

import torch

from config import load_config
from data import get_mnist_loaders
from model import VAE, FlowModel


def main() -> None:
    cfg = load_config(None)
    device = torch.device("cpu")
    vae = VAE(cfg).to(device)
    vae.load_state_dict(torch.load(cfg.paths.vae_checkpoint, map_location=device)["model"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    train_loader, _ = get_mnist_loaders(cfg)

    zs, mus, logvars, labels_all, imgs_all = [], [], [], [], []
    n = 0
    for images, labels in train_loader:
        images = images.to(device)
        mu, logvar = vae.encode(images)
        zs.append(vae.reparameterize(mu, logvar))
        mus.append(mu)
        logvars.append(logvar)
        labels_all.append(labels)
        imgs_all.append(images)  # 与 mu 同序保存，重建对比时必须对齐（loader 是 shuffle 的）
        n += images.shape[0]
        if n >= 1024:
            break

    z1 = torch.cat(zs)
    mu = torch.cat(mus)
    logvar = torch.cat(logvars)
    labels = torch.cat(labels_all)[: z1.shape[0]].to(device)
    images_aligned = torch.cat(imgs_all)[: z1.shape[0]]

    print("latent z1 shape", tuple(z1.shape))
    print("z1   mean %.4f  std %.4f  min %.3f max %.3f"
          % (z1.mean(), z1.std(), z1.min(), z1.max()))
    print("mu   mean %.4f  std %.4f" % (mu.mean(), mu.std()))
    print("posterior std mean %.4f" % logvar.mul(0.5).exp().mean())
    print("per-element variance of z1 (avg) %.4f" % z1.var(dim=0).mean())

    z0 = torch.randn_like(z1)
    target = z1 - z0
    print("\n--- flow MSE baselines ---")
    print("predict 0          MSE = %.4f" % target.pow(2).mean())
    print("predict E[target]  MSE = %.4f" % (target - target.mean(0, keepdim=True)).pow(2).mean())

    flow = FlowModel(cfg).to(device)
    flow.load_state_dict(torch.load(cfg.paths.flow_checkpoint, map_location=device)["model"])
    flow.eval()

    # 若 z1 ~ N(0,I) 且与标签无关，则最优速度场只能利用 z_t 的线性高斯信息，
    # 每元素 MSE 下界 = 2 - (2t-1)^2 / ((1-t)^2 + t^2)
    def theoretical_floor(tval: float) -> float:
        return 2.0 - (2 * tval - 1) ** 2 / ((1 - tval) ** 2 + tval ** 2)

    print("\n--- trained model MSE per t  vs  噪声潜变量理论下界 ---")
    with torch.no_grad():
        for tval in [0.05, 0.25, 0.5, 0.75, 0.95]:
            t = torch.full((z1.shape[0],), tval)
            vt = t.view(-1, 1, 1, 1)
            zt = (1 - vt) * z0 + vt * z1
            pred = flow(zt, t, labels)
            mse = (pred - target).pow(2).mean().item()
            print("t=%.2f  model MSE=%.4f  理论下界=%.4f  pred_std=%.4f"
                  % (tval, mse, theoretical_floor(tval), pred.std()))

    grid = torch.linspace(0.0, 1.0, 200)
    avg_floor = sum(theoretical_floor(float(x)) for x in grid) / len(grid)
    print("\nt 上平均理论下界 ≈ %.4f  (训练 loss 卡在 ~1.557)" % avg_floor)

    # 标签是否真的影响潜变量？比较各类 mu 的类间分离度
    print("\n--- 潜变量是否携带数字信息 ---")
    per_class = []
    for c in range(10):
        m = mu[labels.cpu() == c]
        if m.shape[0] > 0:
            per_class.append(m.mean(0))
    class_means = torch.stack(per_class)
    between = class_means.var(dim=0).mean().item()      # 类间方差
    within = mu.var(dim=0).mean().item()                # 总方差
    print("mu 类间方差 %.5f  /  mu 总方差 %.5f  ->  可解释比例 %.3f"
          % (between, within, between / max(within, 1e-9)))

    # 解码器是否真的用到了潜变量？对比 真实 mu 解码 vs 随机噪声解码 的重建
    print("\n--- 解码器是否依赖潜变量 ---")
    images = images_aligned.to(device)  # 与 mu 严格同序
    with torch.no_grad():
        recon_mu = vae.decode(mu)
        recon_rand = vae.decode(torch.randn_like(mu))
        bce_mu = torch.nn.functional.binary_cross_entropy(recon_mu, images).item()
        bce_rand = torch.nn.functional.binary_cross_entropy(recon_rand, images).item()
    print("BCE 用真实 mu 重建   %.4f" % bce_mu)
    print("BCE 用随机噪声重建   %.4f  (若与上面接近，说明解码器几乎不用潜变量)" % bce_rand)

    # 不受 logit 饱和影响的干净指标：前景像素二值化准确率与 IoU
    tgt_fg = images[:, 1] > 0.5
    pred_fg = recon_mu[:, 1] > 0.5
    acc = (pred_fg == tgt_fg).float().mean().item()
    inter = (pred_fg & tgt_fg).float().sum().item()
    union = (pred_fg | tgt_fg).float().sum().item()
    base_fg = tgt_fg.float().mean().item()
    print("前景像素准确率 %.4f  (全预测背景的基线 %.4f)" % (acc, 1 - base_fg))
    print("前景 IoU       %.4f" % (inter / max(union, 1.0)))


if __name__ == "__main__":
    main()
