# exp1_gradnorm_sgd_vs_adam.py
# 实验题1：绘制前10个epoch的“权重梯度L2范数”曲线，并结合训练集Dice波动进行对比
import os
import argparse
import random
import numpy as np

import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import train_spleen_opt as spleen


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def grad_l2_norm(model: torch.nn.Module) -> float:
    """
    计算一次 backward 之后的“权重梯度L2范数”
    这里取所有参数梯度的整体L2范数：sqrt(sum(||grad||^2))
    """
    s = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        s += float(torch.sum(g * g).item())
    return float(np.sqrt(s))


@torch.no_grad()
def eval_train_dice(model: torch.nn.Module, loader: DataLoader, device: str) -> float:
    """
    在训练集上计算平均 Dice（用于观察波动）
    """
    model.eval()
    dices = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        p = model(x)
        # 复用 train_spleen_opt.py 的 dice_score
        d = spleen.dice_score(p, y).item()
        dices.append(d)
    return float(np.mean(dices)) if len(dices) > 0 else float("nan")


def train_one_optimizer(opt_name: str, data_root: str, epochs: int, lr: float, base: int,
                        batch: int, max_cases: int, seed: int, outdir: str):
    """
    训练一个优化器（SGD 或 Adam），记录每个epoch：
    - 梯度L2范数（按batch平均）
    - 训练集Dice（epoch结束后评估）
    """
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 训练集：按你现有代码逻辑（默认过滤空白切片），用于快速实验
    ds = spleen.Spleen2DDataset(data_root, max_cases=max_cases, include_empty=False)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0)

    # Dice评估同样使用训练集（不做验证/测试，符合实验题1描述）
    dl_eval = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)

    model = spleen.UNetMini(base=base).to(device)

    # 优化器：固定学习率 0.001；SGD 加动量 0.9
    if opt_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif opt_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError("opt_name must be 'sgd' or 'adam'")

    bce = torch.nn.BCELoss()

    records = []  # (epoch, grad_norm_mean, train_dice)

    for ep in range(1, epochs + 1):
        model.train()
        grad_norms = []

        for x, y in dl:
            x, y = x.to(device), y.to(device)
            p = model(x)
            loss = 0.5 * bce(p, y) + 0.5 * spleen.dice_loss(p, y)

            optimizer.zero_grad()
            loss.backward()

            # 记录本batch梯度L2范数（在 step 前记录即可）
            gn = grad_l2_norm(model)
            grad_norms.append(gn)

            optimizer.step()

        grad_norm_mean = float(np.mean(grad_norms)) if len(grad_norms) > 0 else float("nan")
        train_dice = eval_train_dice(model, dl_eval, device)

        records.append((ep, grad_norm_mean, train_dice))
        print(f"[{opt_name.upper()}] EP{ep:02d} gradL2={grad_norm_mean:.6f} trainDice={train_dice:.4f}")

    # 保存CSV
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, f"exp1_{opt_name.lower()}_metrics.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("epoch,grad_l2_norm_mean,train_dice\n")
        for ep, gn, dc in records:
            f.write(f"{ep},{gn:.8f},{dc:.6f}\n")
    print("[SAVE]", os.path.abspath(csv_path))

    return records


def plot_curves(records_sgd, records_adam, outdir: str):
    """
    画两张曲线（同一张图两个子图）：
    - 上：权重梯度L2范数（epoch均值）
    - 下：训练集Dice
    """
    os.makedirs(outdir, exist_ok=True)

    e = [r[0] for r in records_sgd]
    sgd_g = [r[1] for r in records_sgd]
    sgd_d = [r[2] for r in records_sgd]

    e2 = [r[0] for r in records_adam]
    adam_g = [r[1] for r in records_adam]
    adam_d = [r[2] for r in records_adam]

    fig = plt.figure(figsize=(9, 6))

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(e, sgd_g, marker="o", label="SGD(m=0.9)")
    ax1.plot(e2, adam_g, marker="o", label="Adam")
    ax1.set_title("Gradient L2 Norm (mean per epoch)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("L2 Norm")
    ax1.legend()

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(e, sgd_d, marker="o", label="SGD(m=0.9)")
    ax2.plot(e2, adam_d, marker="o", label="Adam")
    ax2.set_title("Training Dice (mean)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Dice")
    ax2.legend()

    plt.tight_layout()
    out_png = os.path.join(outdir, "exp1_gradnorm_and_dice.png")
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("[SAVE]", os.path.abspath(out_png))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to Task09_Spleen")
    ap.add_argument("--epochs", type=int, default=10, help="Use first N epochs (default 10)")
    ap.add_argument("--lr", type=float, default=1e-3, help="Fixed learning rate (default 0.001)")
    ap.add_argument("--base", type=int, default=16, help="UNet base channels (must match your spleen code)")
    ap.add_argument("--batch", type=int, default=4, help="Batch size")
    ap.add_argument("--max_cases", type=int, default=3, help="Use first K cases for a lightweight run")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default=os.path.join("data", "reports", "exp1"),
                    help="Output directory for CSV and PNG")
    args = ap.parse_args()

    print("[INFO] Experiment 1: SGD(m=0.9) vs Adam | first 10 epochs | fixed lr=0.001")
    print("[INFO] Output ->", os.path.abspath(args.outdir))

    rec_sgd = train_one_optimizer(
        opt_name="sgd",
        data_root=args.data,
        epochs=args.epochs,
        lr=args.lr,
        base=args.base,
        batch=args.batch,
        max_cases=args.max_cases,
        seed=args.seed,
        outdir=args.outdir
    )

    rec_adam = train_one_optimizer(
        opt_name="adam",
        data_root=args.data,
        epochs=args.epochs,
        lr=args.lr,
        base=args.base,
        batch=args.batch,
        max_cases=args.max_cases,
        seed=args.seed,
        outdir=args.outdir
    )

    plot_curves(rec_sgd, rec_adam, args.outdir)


if __name__ == "__main__":
    main()
