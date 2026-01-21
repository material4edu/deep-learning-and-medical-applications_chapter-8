import os, glob, argparse
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import nibabel as nib
import cv2
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'SimHei'

# -------------------------
# Model (must match training architecture)
# -------------------------
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

    def forward(self, x):
        return self.net(x)


class UNetMini(nn.Module):
    def __init__(self, base=32, p_drop=0.0):
        super().__init__()
        self.d1 = DoubleConv(1, base, p_drop);      self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(base, base*2, p_drop); self.p2 = nn.MaxPool2d(2)
        self.b  = DoubleConv(base*2, base*4, p_drop)
        self.u2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.u1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.c2 = DoubleConv(base*4 + base*2, base*2, p_drop)
        self.c1 = DoubleConv(base*2 + base, base, p_drop)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        x1 = self.d1(x)
        x2 = self.d2(self.p1(x1))
        xb = self.b(self.p2(x2))
        y2 = self.c2(torch.cat([self.u2(xb), x2], 1))
        y1 = self.c1(torch.cat([self.u1(y2), x1], 1))
        return torch.sigmoid(self.out(y1))


# -------------------------
# IO helpers
# -------------------------
def load_nii(path: str) -> np.ndarray:
    return np.asarray(nib.load(path).get_fdata())


def zscore_volume(vol: np.ndarray) -> np.ndarray:
    vol = vol.astype(np.float32)
    m = float(vol.mean())
    s = float(vol.std()) + 1e-6
    return (vol - m) / s


def binarize_mask(m: np.ndarray, thr: float = 0.5) -> np.ndarray:
    m = m.astype(np.float32)
    if m.max() <= 1.0 and m.min() >= 0.0:
        return (m >= thr).astype(np.uint8)
    return (m > 0).astype(np.uint8)


def normalize_bg(x2d: np.ndarray, p_low=1, p_high=99) -> np.ndarray:
    x = x2d.astype(np.float32)
    vmin = np.percentile(x, p_low)
    vmax = np.percentile(x, p_high)
    x = np.clip((x - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
    return x


def list_image_files(images_dir: str):
    # Prefer *_0000.nii.gz if present
    p1 = sorted(glob.glob(os.path.join(images_dir, "*_0000.nii.gz")))
    if len(p1) > 0:
        return p1
    p2 = sorted(glob.glob(os.path.join(images_dir, "*.nii.gz")))
    return p2


def infer_case_id(img_path: str):
    name = os.path.basename(img_path)
    if name.endswith("_0000.nii.gz"):
        return name.replace("_0000.nii.gz", "")
    if name.endswith(".nii.gz"):
        return name.replace(".nii.gz", "")
    return os.path.splitext(os.path.splitext(name)[0])[0]


def find_case_paths(root: str, split: str, case_id: str):
    # split: "Tr" or "Ts"
    img_dir = os.path.join(root, f"images{split}")
    lab_dir = os.path.join(root, "labelsTr")  # only labelsTr exists for MSD

    cand_img1 = os.path.join(img_dir, f"{case_id}.nii.gz")
    cand_img2 = os.path.join(img_dir, f"{case_id}_0000.nii.gz")
    if os.path.exists(cand_img2):
        img_path = cand_img2
    elif os.path.exists(cand_img1):
        img_path = cand_img1
    else:
        cands = glob.glob(os.path.join(img_dir, f"{case_id}*.nii.gz"))
        if len(cands) == 0:
            raise FileNotFoundError(f"Cannot find image for {case_id} in {img_dir}")
        img_path = sorted(cands)[0]

    lab_path = os.path.join(lab_dir, f"{case_id}.nii.gz")
    if not os.path.exists(lab_path):
        lab_path = None  # test split has no labels
    return img_path, lab_path


def auto_pick_case_with_gt(root: str):
    # Choose case with max foreground voxels in labelsTr
    lab_dir = os.path.join(root, "labelsTr")
    lab_paths = sorted(glob.glob(os.path.join(lab_dir, "*.nii.gz")))
    if len(lab_paths) == 0:
        raise FileNotFoundError(f"No labels found in {lab_dir}")
    best_id, best_sum = None, -1
    for lp in lab_paths:
        gt = binarize_mask(load_nii(lp))
        s = int(gt.sum())
        if s > best_sum:
            best_sum = s
            best_id = os.path.basename(lp).replace(".nii.gz", "")
    return best_id


def find_best_slice(gt3d_256: np.ndarray) -> int:
    areas = [int((gt3d_256[:, :, z] > 0).sum()) for z in range(gt3d_256.shape[2])]
    return int(np.argmax(areas))


def mask_to_contour(mask2d: np.ndarray, thickness: int = 2) -> np.ndarray:
    m = (mask2d > 0).astype(np.uint8)
    if m.sum() == 0:
        return np.zeros_like(m, dtype=np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    canvas = np.zeros_like(m, dtype=np.uint8)
    cv2.drawContours(canvas, contours, -1, color=1, thickness=thickness)
    return canvas


@torch.no_grad()
def predict_volume_slicewise(net, img3d_z, thr=0.5, device="cpu"):
    # Predict on resized 256x256 slices and return a (256,256,Z) binary volume
    net.eval()
    preds = []
    for z in range(img3d_z.shape[2]):
        x2d = cv2.resize(img3d_z[:, :, z].astype(np.float32), (256, 256)).astype(np.float32)
        x = torch.from_numpy(x2d[None, None, ...]).to(device)  # (1,1,H,W)
        p = net(x).squeeze().detach().cpu().numpy()
        preds.append((p >= thr).astype(np.uint8))
    return np.stack(preds, axis=2)


def build_gt_256(gt3d: np.ndarray) -> np.ndarray:
    # Resize each GT slice to 256x256
    gt3d = binarize_mask(gt3d)
    outs = []
    for z in range(gt3d.shape[2]):
        m = cv2.resize(gt3d[:, :, z].astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST)
        outs.append((m > 0).astype(np.uint8))
    return np.stack(outs, axis=2)


def save_triplet(ct_norm, pred2d, gt_contour, out_path, title_prefix=""):
    # Panel 1: CT
    # Panel 2: CT + Pred(mask)
    # Panel 3: CT + Pred(mask) + GT(contour)
    pred_rgba = np.zeros((ct_norm.shape[0], ct_norm.shape[1], 4), dtype=np.float32)
    pred_rgba[..., 0] = 1.0
    pred_rgba[..., 3] = 0.35 * pred2d

    gt_rgba = np.zeros((ct_norm.shape[0], ct_norm.shape[1], 4), dtype=np.float32)
    gt_rgba[..., 1] = 1.0
    gt_rgba[..., 3] = 0.95 * gt_contour

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(ct_norm, cmap="gray")
    axes[0].set_title(f"{title_prefix}CT".strip(), fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(ct_norm, cmap="gray")
    axes[1].imshow(pred_rgba)
    axes[1].set_title(f"{title_prefix}CT + 预测掩膜".strip(), fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(ct_norm, cmap="gray")
    axes[2].imshow(pred_rgba)
    axes[2].imshow(gt_rgba)
    axes[2].set_title(f"{title_prefix}CT + 预测掩膜 + 标注轮廓".strip(), fontsize=12)
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def save_double(ct_norm, pred2d, out_path, title_prefix=""):
    # Panel 1: CT
    # Panel 2: CT + Pred(mask)
    pred_rgba = np.zeros((ct_norm.shape[0], ct_norm.shape[1], 4), dtype=np.float32)
    pred_rgba[..., 0] = 1.0
    pred_rgba[..., 3] = 0.35 * pred2d

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(ct_norm, cmap="gray")
    axes[0].set_title(f"{title_prefix}CT".strip(), fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(ct_norm, cmap="gray")
    axes[1].imshow(pred_rgba)
    axes[1].set_title(f"{title_prefix}CT + Pred(mask)".strip(), fontsize=12)
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to Task05_Prostate or Task09_Spleen")
    ap.add_argument("--weights", required=True, help="Path to model .pth")
    ap.add_argument("--split", default="Tr", choices=["Tr", "Ts"], help="Use imagesTr or imagesTs")
    ap.add_argument("--case", default="", help="Case id like prostate_16 or spleen_6; empty => auto pick (Tr only)")
    ap.add_argument("--slice", type=int, default=-1, help="Slice index z; -1 => auto pick")
    ap.add_argument("--pred_thr", type=float, default=0.5, help="Threshold for predicted probability")
    ap.add_argument("--base", type=int, default=32, help="UNet base channels used in training")
    ap.add_argument("--outdir", default=os.path.join("data", "reports"), help="Output directory")
    ap.add_argument("--tag", default="", help="Extra tag for filename")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = UNetMini(base=args.base, p_drop=0.0).to(device)
    state = torch.load(args.weights, map_location=device)
    net.load_state_dict(state, strict=False)

    # Determine case id
    if args.case.strip() == "":
        if args.split != "Tr":
            raise ValueError("Auto case selection requires --split Tr because test split has no GT.")
        case_id = auto_pick_case_with_gt(args.data)
    else:
        case_id = args.case.strip()

    img_path, lab_path = find_case_paths(args.data, args.split, case_id)

    img = load_nii(img_path).astype(np.float32)
    if img.ndim == 4:
        img = img[..., 0]

    img_z = zscore_volume(img)
    pred_256 = predict_volume_slicewise(net, img_z, thr=args.pred_thr, device=device)

    # Determine z slice
    if lab_path is not None:
        gt = load_nii(lab_path)
        gt_256 = build_gt_256(gt)
        z = args.slice if args.slice >= 0 else find_best_slice(gt_256)
        gt2d = gt_256[:, :, z].astype(np.uint8)
        gt_contour = mask_to_contour(gt2d, thickness=2).astype(np.float32)
    else:
        z = args.slice if args.slice >= 0 else int(pred_256.shape[2] // 2)
        gt_contour = None

    # Background CT slice in 256 space
    ct2d = cv2.resize(img_z[:, :, z].astype(np.float32), (256, 256)).astype(np.float32)
    ct_norm = normalize_bg(ct2d)

    pred2d = pred_256[:, :, z].astype(np.float32)

    tag = f"_{args.tag}" if args.tag.strip() else ""
    if lab_path is not None:
        out_name = f"triplet_overlay_{case_id}_z{z}{tag}.png"
        out_path = os.path.join(args.outdir, out_name)
        save_triplet(ct_norm, pred2d, gt_contour, out_path, title_prefix="")
    else:
        out_name = f"double_overlay_{case_id}_z{z}{tag}.png"
        out_path = os.path.join(args.outdir, out_name)
        save_double(ct_norm, pred2d, out_path, title_prefix="")

    print("[OK] split:", args.split)
    print("[OK] case:", case_id)
    print("[OK] z:", z)
    print("[OK] image:", os.path.abspath(img_path))
    if lab_path is not None:
        print("[OK] label:", os.path.abspath(lab_path))
    print("[OK] weights:", os.path.abspath(args.weights))
    print("[OK] saved:", os.path.abspath(out_path))


if __name__ == "__main__":
    main()
