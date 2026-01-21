import os, glob, argparse
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import nibabel as nib
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")  # 不弹窗、直接存PNG，尤其在VSCode/远程/无界面下更稳
import matplotlib.pyplot as plt

class DoubleConv(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c1, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)

class UNetMini(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        self.d1=DoubleConv(1, base);      self.p1=nn.MaxPool2d(2)
        self.d2=DoubleConv(base, base*2); self.p2=nn.MaxPool2d(2)
        self.b =DoubleConv(base*2, base*4)
        self.u2=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.u1=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.c2=DoubleConv(base*4+base*2, base*2)
        self.c1=DoubleConv(base*2+base, base)
        self.out=nn.Conv2d(base, 1, 1)
    def forward(self, x):
        x1=self.d1(x); x2=self.d2(self.p1(x1)); xb=self.b(self.p2(x2))
        y2=self.c2(torch.cat([self.u2(xb), x2],1))
        y1=self.c1(torch.cat([self.u1(y2), x1],1))
        return torch.sigmoid(self.out(y1))

class Spleen2DDataset(Dataset):
    def __init__(self, root, max_cases=None, include_empty=False):
        img_paths = sorted(glob.glob(os.path.join(root, "imagesTr", "*.nii.gz")))
        if len(img_paths) == 0:
            raise FileNotFoundError(f"No images found at {os.path.join(root,'imagesTr','*.nii.gz')}")
        if max_cases is not None:
            img_paths = img_paths[:max_cases]
        self.pairs = []
        for ip in img_paths:
            base = os.path.basename(ip).replace(".nii.gz", ".nii.gz")
            lp = os.path.join(root, "labelsTr", base)
            if not os.path.exists(lp):
                print(f"[WARN] Label not found for {base}, skip.")
                continue
            img = nib.load(ip).get_fdata().astype(np.float32)
            lab = nib.load(lp).get_fdata().astype(np.uint8)

            # 简易窗宽窗位：[-100,400] → [0,1]
            img = np.clip((img + 100) / 500, 0, 1)

            for z in range(img.shape[2]):
                i = cv2.resize(img[:, :, z], (256, 256)).astype(np.float32)
                m = cv2.resize(lab[:, :, z], (256, 256), interpolation=cv2.INTER_NEAREST)
                m = (m > 0).astype(np.float32)
                if not include_empty:
                    # 训练集默认过滤“几乎全空白”的切片
                    if m.sum() < 10:
                        continue
                self.pairs.append((i, m))
        if len(self.pairs) == 0:
            raise RuntimeError("No slices built. Check dataset path or filtering threshold.")

    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        i, m = self.pairs[idx]
        return torch.from_numpy(i[None, ...]), torch.from_numpy(m[None, ...])

def dice_loss(pred, target, eps=1e-6):
    num = 2 * (pred * target).sum()
    den = pred.sum() + target.sum() + eps
    return 1 - num / den

@torch.no_grad()
def dice_score(p, y, eps=1e-6):
    p = (p > 0.5).float()
    inter = (p * y).sum()
    return (2 * inter) / (p.sum() + y.sum() + eps)

@torch.no_grad()
def iou_score(p, y, eps=1e-6):
    p = (p > 0.5).float()
    inter = (p * y).sum()
    union = p.sum() + y.sum() - inter
    return inter / (union + eps)

try:
    from scipy.ndimage import distance_transform_edt
    _has_scipy = True
except Exception:
    _has_scipy = False

@torch.no_grad()
def hd95_score(p, y):
    if not _has_scipy:
        return float("nan")
    p = (p > 0.5).float().squeeze().cpu().numpy()
    y = (y > 0.5).float().squeeze().cpu().numpy()

    def _surface_dt(a, b):
        dt = distance_transform_edt(1 - b)
        vals = dt[a.astype(bool)]
        if vals.size == 0: return np.array([0.0])
        return vals

    d1 = _surface_dt(p, y)
    d2 = _surface_dt(y, p)
    return float(np.percentile(np.hstack([d1, d2]), 95))

def get_opt(params, name, lr):
    name = name.lower()
    if name == "sgd":     return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if name == "rmsprop": return torch.optim.RMSprop(params, lr=lr, alpha=0.99)
    return torch.optim.Adam(params, lr=lr)  # 默认 Adam

@torch.no_grad()
def evaluate_models_for_report(ds, model_paths, base=16, device="cpu"):
    net = UNetMini(base=base).to(device)
    rows = []
    for mp in model_paths:
        if not os.path.exists(mp):
            print(f"[WARN] {mp} not found, skip."); continue
        state = torch.load(mp, map_location=device)
        net.load_state_dict(state, strict=True)
        net.eval()
        dices, ious, hd95s = [], [], []
        for k in range(len(ds)):
            x, y = ds[k]
            x, y = x.to(device).unsqueeze(0), y.to(device).unsqueeze(0)
            p = net(x)
            dices.append(dice_score(p, y).item())
            ious.append(iou_score(p, y).item())
            hd95s.append(hd95_score(p, y))
        m_dice = float(np.mean(dices))
        m_iou  = float(np.mean(ious))
        m_hd95 = float(np.nanmean(hd95s)) if not np.isnan(hd95s).all() else float("nan")
        name   = os.path.basename(mp)
        print(f"[REPORT] {name}: Dice={m_dice:.4f} IoU={m_iou:.4f} HD95={m_hd95 if not np.isnan(m_hd95) else 'NA'}")
        rows.append((name, m_dice, m_iou, m_hd95))
    return rows

def save_report_csv(path, rows):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model","dice","iou","hd95_pix"])
        for r in rows: w.writerow(r)
    print(f"[SAVE] {path}")

def plot_report_bars(path, rows, title):
    names = [r[0] for r in rows]
    dice  = [r[1] for r in rows]
    iou   = [r[2] for r in rows]
    hd95  = [0 if (isinstance(r[3], float) and np.isnan(r[3])) else r[3] for r in rows]
    x = np.arange(len(names)); w = 0.25
    plt.figure(figsize=(10,5))
    plt.bar(x - w, dice, w, label="Dice")
    plt.bar(x,     iou,  w, label="IoU")
    plt.bar(x + w, hd95, w, label="HD95 (pix)")
    plt.xticks(x, names, rotation=20, ha="right")
    plt.ylabel("Score / Pixels"); plt.title(title)
    plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()
    print(f"[SAVE] {path}")

# ------------------------
# 主流程
# ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to MSD Task09_Spleen")
    ap.add_argument("--opt", default="adam", choices=["adam","sgd","rmsprop"], help="optimizer")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--base", type=int, default=16, help="base channels in UNet")
    ap.add_argument("--max_cases", type=int, default=3, help="how many cases for quick training")
    # 报告模式
    ap.add_argument("--report", action="store_true", help="Run report mode only")
    ap.add_argument("--models", nargs="+", help="Model weights for report mode")
    args = ap.parse_args()

    os.makedirs("data/models", exist_ok=True)
    os.makedirs("data/reports", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ========= 报告模式：不训练，直接评估多个权重并生成教材图表 =========
    if args.report:
        ds_full = Spleen2DDataset(args.data, max_cases=None, include_empty=True)  # 用全量切片
        rows = evaluate_models_for_report(ds_full, args.models or [], base=args.base, device=device)
        if len(rows) == 0:
            print("[REPORT] No valid models to evaluate."); return
        save_report_csv("data/reports/report_spleen.csv", rows)
        plot_report_bars("data/reports/report_spleen_metrics.png", rows, "Spleen - Optimizer Comparison")
        return

    # ========= 训练模式 =========
    ds = Spleen2DDataset(args.data, max_cases=args.max_cases, include_empty=False)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)
    net = UNetMini(base=args.base).to(device)
    opt = get_opt(net.parameters(), args.opt, args.lr)
    bce = nn.BCELoss()

    losses_per_epoch = []
    for ep in range(1, args.epochs + 1):
        net.train(); losses=[]
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            p = net(x)
            loss = 0.5 * bce(p, y) + 0.5 * dice_loss(p, y)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        mean_loss = float(np.mean(losses))
        losses_per_epoch.append(mean_loss)
        print(f"[EP{ep}] mean loss={mean_loss:.4f}")

    # 保存权重
    wt_path = f"data/models/spleen_{args.opt}.pth"
    torch.save(net.state_dict(), wt_path)
    print(f"[SAVE] {wt_path}")

    # 保存训练曲线 CSV + PNG
    import csv, matplotlib.pyplot as plt
    losses_csv = f"data/reports/spleen_{args.opt}_loss.csv"
    with open(losses_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["epoch","mean_loss"])
        for i, v in enumerate(losses_per_epoch, 1):
            w.writerow([i, float(v)])
    print(f"[SAVE] {losses_csv}")

    plt.figure(figsize=(6,4))
    plt.plot(range(1, len(losses_per_epoch)+1), losses_per_epoch, marker="o")
    plt.xlabel("Epoch"); plt.ylabel("Mean Loss"); plt.title(f"Spleen - {args.opt} Loss")
    plt.tight_layout(); plt.savefig(f"data/reports/spleen_{args.opt}_loss.png", dpi=150); plt.close()
    print(f"[SAVE] reports/spleen_{args.opt}_loss.png")

if __name__ == "__main__":
    main()