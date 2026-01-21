# train_prostate_reg.py
import os, glob, argparse
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import nibabel as nib
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------
# Model
# ------------------------
class DoubleConv(nn.Module):
    def __init__(self, c1, c2, p_drop=0.0):
        super().__init__()
        mods = [
            nn.Conv2d(c1, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(inplace=True)
        ]
        if p_drop > 0:
            mods.append(nn.Dropout(p_drop))
        self.net = nn.Sequential(*mods)
    def forward(self, x): return self.net(x)

class UNetMini(nn.Module):
    def __init__(self, base=32, p_drop=0.0):
        super().__init__()
        self.d1 = DoubleConv(1, base, p_drop);      self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(base, base*2, p_drop); self.p2 = nn.MaxPool2d(2)
        self.b  = DoubleConv(base*2, base*4, p_drop)
        self.u2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.u1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.c2 = DoubleConv(base*4+base*2, base*2, p_drop)
        self.c1 = DoubleConv(base*2+base, base, p_drop)
        self.out = nn.Conv2d(base, 1, 1)
    def forward(self, x):
        x1 = self.d1(x); x2 = self.d2(self.p1(x1)); xb = self.b(self.p2(x2))
        y2 = self.c2(torch.cat([self.u2(xb), x2], 1))
        y1 = self.c1(torch.cat([self.u1(y2), x1], 1))
        return torch.sigmoid(self.out(y1))

# ------------------------
# Dataset
# ------------------------
class Prostate2DDataset(Dataset):
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
            if img.ndim == 4:
                img = img[..., 0]  # use first modality
            lab = nib.load(lp).get_fdata().astype(np.uint8)

            m, s = img.mean(), img.std() + 1e-6
            img = (img - m) / s

            for z in range(img.shape[2]):
                i = cv2.resize(img[:, :, z], (256, 256)).astype(np.float32)
                msk = cv2.resize(lab[:, :, z], (256, 256), interpolation=cv2.INTER_NEAREST)
                msk = (msk > 0).astype(np.float32)  # binarize {0,1,2} -> {0,1}
                if not include_empty and msk.sum() < 10:
                    continue
                self.pairs.append((i, msk))
        if len(self.pairs) == 0:
            raise RuntimeError("No slices built. Check dataset path or filtering threshold.")

    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        i, m = self.pairs[idx]
        return torch.from_numpy(i[None, ...]), torch.from_numpy(m[None, ...])

# ------------------------
# Loss & Metrics
# ------------------------
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

# ------------------------
# Optimizer
# ------------------------
def get_opt(params, name, lr, weight_decay=0.0):
    name = name.lower()
    if name == "sgd":     return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "rmsprop": return torch.optim.RMSprop(params, lr=lr, alpha=0.99, weight_decay=weight_decay)
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

# ------------------------
# Report helpers
# ------------------------
@torch.no_grad()
def evaluate_models_for_report(ds, model_paths, base=32, device="cpu"):
    net = UNetMini(base=base, p_drop=0.0).to(device)
    rows = []
    for mp in model_paths:
        if not os.path.exists(mp):
            print(f"[WARN] {mp} not found, skip."); continue
        state = torch.load(mp, map_location=device)
        net.load_state_dict(state, strict=False)
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
# Main
# ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to MSD Task05_Prostate")
    ap.add_argument("--reg",  default="baseline", choices=["baseline","dropout","l2"])
    ap.add_argument("--opt",  default="adam", choices=["adam","sgd","rmsprop"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--max_cases", type=int, default=3)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--models", nargs="+")
    args = ap.parse_args()

    os.makedirs("data/models", exist_ok=True)
    os.makedirs("data/reports", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.report:
        ds_full = Prostate2DDataset(args.data, max_cases=None, include_empty=True)
        rows = evaluate_models_for_report(ds_full, args.models or [], base=args.base, device=device)
        if len(rows) == 0:
            print("[REPORT] No valid models to evaluate."); return
        save_report_csv("data/reports/report_prostate.csv", rows)
        plot_report_bars("data/reports/report_prostate_metrics.png", rows, "Prostate - Regularization Comparison")
        return

    p_drop = 0.3 if args.reg == "dropout" else 0.0
    net = UNetMini(base=args.base, p_drop=p_drop).to(device)
    weight_decay = 1e-4 if args.reg == "l2" else 0.0
    opt = get_opt(net.parameters(), args.opt, args.lr, weight_decay=weight_decay)
    bce = nn.BCELoss()

    ds = Prostate2DDataset(args.data, max_cases=args.max_cases, include_empty=False)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)

    losses_per_epoch = []
    for ep in range(1, args.epochs + 1):
        net.train(); losses = []
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            p = net(x)
            loss = 0.5 * bce(p, y) + 0.5 * dice_loss(p, y)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        mean_loss = float(np.mean(losses))
        losses_per_epoch.append(mean_loss)
        print(f"[EP{ep}] mean loss={mean_loss:.4f}")

    wt_path = f"data/models/prostate_{args.reg}.pth"
    torch.save(net.state_dict(), wt_path)
    print(f"[SAVE] {wt_path}")

    import csv
    losses_csv = f"data/reports/prostate_{args.reg}_loss.csv"
    with open(losses_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["epoch","mean_loss"])
        for i, v in enumerate(losses_per_epoch, 1):
            w.writerow([i, float(v)])
    print(f"[SAVE] {losses_csv}")

    plt.figure(figsize=(6,4))
    plt.plot(range(1, len(losses_per_epoch)+1), losses_per_epoch, marker="o")
    plt.xlabel("Epoch"); plt.ylabel("Mean Loss"); plt.title(f"Prostate - {args.reg} Loss")
    plt.tight_layout()
    plt.savefig(f"data/reports/prostate_{args.reg}_loss.png", dpi=150)
    plt.close()
    print(f"[SAVE] data/reports/prostate_{args.reg}_loss.png")

if __name__ == "__main__":
    main()