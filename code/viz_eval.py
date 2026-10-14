import os, glob, argparse
import json
from pathlib import Path
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
    """Apply the volume normalization used by the baseline models."""
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
    """Extract external contours from contiguous or reoriented binary masks."""
    m = np.ascontiguousarray(mask2d > 0, dtype=np.uint8)
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


def prepare_case_slice(root, split, case_id, slice_index, net,
                       normalization, threshold=0.5, device="cpu"):
    """Predict one native axial slice and align display layers using its affine."""
    if normalization not in ("zscore", "ct_window"):
        raise ValueError("Normalization must be zscore or ct_window.")
    if not 0.0 < threshold < 1.0:
        raise ValueError("Prediction threshold must be between zero and one.")
    metadata = json.loads((Path(root) / "dataset.json").read_text(encoding="utf-8"))
    modality = metadata["modality"]["0"]
    if normalization == "ct_window" and modality != "CT":
        raise ValueError("CT window normalization requires CT data.")
    image_path, label_path = find_case_paths(root, split, case_id)
    image = nib.load(image_path)
    axes = nib.aff2axcodes(image.affine)
    if axes[2] not in ("S", "I"):
        raise ValueError("This axial slice model requires an axial source volume.")
    raw = np.asarray(image.dataobj, dtype=np.float32)
    if raw.ndim == 4:
        raw = raw[..., 0]
    if raw.ndim != 3:
        raise ValueError("Expected a 3D image after selecting the first modality.")
    label = None
    gt_256 = None
    if label_path is not None:
        label_image = nib.load(label_path)
        if label_image.shape != raw.shape or not np.allclose(label_image.affine, image.affine):
            raise ValueError("Image and annotation must share the same spatial grid.")
        label = np.asarray(label_image.dataobj)
        gt_256 = build_gt_256(label)
    z = slice_index
    if z < 0:
        z = find_best_slice(gt_256) if gt_256 is not None else raw.shape[2] // 2
    if not 0 <= z < raw.shape[2]:
        raise ValueError("Slice index is outside the source volume.")

    if normalization == "zscore":
        model_volume = zscore_volume(raw)
    else:
        # Match train_spleen_opt.py: map [-100, 400] HU to [0, 1].
        model_volume = np.clip((raw + 100.0) / 500.0, 0.0, 1.0)
    prediction = predict_volume_slicewise(
        net, model_volume[:, :, z:z + 1], thr=threshold, device=device,
    )[:, :, 0]
    background = cv2.resize(raw[:, :, z], (256, 256))
    if modality == "CT":
        background = np.clip((background + 100.0) / 500.0, 0.0, 1.0)
    else:
        background = normalize_bg(background)

    # Rows run anterior-to-posterior; columns run patient-right-to-left.
    # Reorient every display layer identically after native-grid inference.
    orientation = nib.orientations.ornt_transform(
        nib.orientations.io_orientation(image.affine),
        nib.orientations.axcodes2ornt(("P", "L", "S")),
    )
    background = nib.orientations.apply_orientation(background[:, :, None], orientation)[:, :, 0]
    prediction = nib.orientations.apply_orientation(prediction[:, :, None], orientation)[:, :, 0]
    target = None
    contour = None
    if gt_256 is not None:
        target = nib.orientations.apply_orientation(gt_256[:, :, z:z + 1], orientation)[:, :, 0]
        contour = mask_to_contour(target, thickness=2)
    scaled_spacing = np.asarray(image.header.get_zooms()[:3], dtype=float)
    scaled_spacing[:2] *= np.asarray(raw.shape[:2]) / 256.0
    display_spacing = scaled_spacing[np.argsort(orientation[:, 0])]
    return {
        "background": background, "prediction": prediction,
        "target": target, "contour": contour,
        "aspect": float(display_spacing[0] / display_spacing[1]),
        "slice": int(z), "modality": modality,
        "source_shape": list(raw.shape), "source_axes": list(axes),
        "label_values": np.unique(label).tolist() if label is not None else [],
    }


def save_triplet(ct_norm, pred2d, gt_contour, out_path, title_prefix="", aspect=1.0):
    """Save neutral titles, a green prediction, and a red annotation contour."""
    pred_rgba = np.zeros((ct_norm.shape[0], ct_norm.shape[1], 4), dtype=np.float32)
    pred_rgba[..., 1] = 1.0
    pred_rgba[..., 3] = 0.35 * pred2d

    gt_rgba = np.zeros((ct_norm.shape[0], ct_norm.shape[1], 4), dtype=np.float32)
    gt_rgba[..., 0] = 1.0
    gt_rgba[..., 3] = 0.95 * gt_contour

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(ct_norm, cmap="gray", vmin=0.0, vmax=1.0, aspect=aspect)
    axes[0].set_title(f"{title_prefix}a. \u539f\u59cb\u56fe\u50cf".strip(), fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(ct_norm, cmap="gray", vmin=0.0, vmax=1.0, aspect=aspect)
    axes[1].imshow(pred_rgba, interpolation="nearest", aspect=aspect)
    axes[1].set_title(f"{title_prefix}b. \u539f\u59cb\u56fe\u50cf + \u9884\u6d4b\u63a9\u819c".strip(), fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(ct_norm, cmap="gray", vmin=0.0, vmax=1.0, aspect=aspect)
    axes[2].imshow(pred_rgba, interpolation="nearest", aspect=aspect)
    axes[2].imshow(gt_rgba, interpolation="nearest", aspect=aspect)
    axes[2].set_title(f"{title_prefix}c. \u539f\u59cb\u56fe\u50cf + \u9884\u6d4b\u63a9\u819c + \u6807\u6ce8\u8f6e\u5ed3".strip(), fontsize=12)
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def save_double(ct_norm, pred2d, out_path, title_prefix="", aspect=1.0):
    """Save neutral image and prediction panels using a green overlay."""
    pred_rgba = np.zeros((ct_norm.shape[0], ct_norm.shape[1], 4), dtype=np.float32)
    pred_rgba[..., 1] = 1.0
    pred_rgba[..., 3] = 0.35 * pred2d

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(ct_norm, cmap="gray", vmin=0.0, vmax=1.0, aspect=aspect)
    axes[0].set_title(f"{title_prefix}a. \u539f\u59cb\u56fe\u50cf".strip(), fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(ct_norm, cmap="gray", vmin=0.0, vmax=1.0, aspect=aspect)
    axes[1].imshow(pred_rgba, interpolation="nearest", aspect=aspect)
    axes[1].set_title(f"{title_prefix}b. \u539f\u59cb\u56fe\u50cf + \u9884\u6d4b\u63a9\u819c".strip(), fontsize=12)
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def main():
    """Render one verified case with explicitly selected model normalization."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to Task05_Prostate or Task09_Spleen")
    ap.add_argument("--weights", required=True, help="Path to model .pth")
    ap.add_argument("--normalization", required=True, choices=["zscore", "ct_window"],
                    help="Training input: baseline uses zscore; spleen optimizer training uses ct_window")
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
    state = torch.load(args.weights, map_location=device, weights_only=True)
    net.load_state_dict(state, strict=True)

    # Determine case id
    if args.case.strip() == "":
        if args.split != "Tr":
            raise ValueError("Auto case selection requires --split Tr because test split has no GT.")
        case_id = auto_pick_case_with_gt(args.data)
    else:
        case_id = args.case.strip()

    result = prepare_case_slice(
        args.data, args.split, case_id, args.slice, net,
        normalization=args.normalization, threshold=args.pred_thr, device=device,
    )
    z = result["slice"]

    tag = f"_{args.tag}" if args.tag.strip() else ""
    if result["contour"] is not None:
        out_name = f"triplet_overlay_{case_id}_z{z}{tag}.png"
        out_path = os.path.join(args.outdir, out_name)
        save_triplet(result["background"], result["prediction"], result["contour"],
                     out_path, aspect=result["aspect"])
    else:
        out_name = f"double_overlay_{case_id}_z{z}{tag}.png"
        out_path = os.path.join(args.outdir, out_name)
        save_double(result["background"], result["prediction"], out_path,
                    aspect=result["aspect"])

    print("[OK] split:", args.split)
    print("[OK] case:", case_id)
    print("[OK] z:", z)
    print("[OK] modality:", result["modality"])
    print("[OK] normalization:", args.normalization)
    print("[OK] weights:", Path(args.weights).name)
    print("[OK] saved:", Path(out_path).name)


if __name__ == "__main__":
    main()
