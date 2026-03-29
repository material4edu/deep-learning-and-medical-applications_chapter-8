# code/exp2_prostate_hd95_heatmap.py
# 实验题2：baseline / dropout / l2 的测试集HD95趋势 + 同一样本边界误差热力图

import os, argparse, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.ndimage import distance_transform_edt
    _has_scipy = True
except Exception:
    _has_scipy = False

import torch

# 复用你现成的可视化/推理工具（包含UNetMini与预处理/推理/轮廓等）
import viz_eval as V


def split_cases(case_ids, seed=42):
    # case-level 70/15/15
    case_ids = list(case_ids)
    rnd = random.Random(seed)
    rnd.shuffle(case_ids)

    n = len(case_ids)
    n_train = int(round(n * 0.70))
    n_val = int(round(n * 0.15))
    train_ids = case_ids[:n_train]
    val_ids = case_ids[n_train:n_train + n_val]
    test_ids = case_ids[n_train + n_val:]
    return train_ids, val_ids, test_ids


def hd95_2d(pred2d: np.ndarray, gt2d: np.ndarray) -> float:
    # 2D HD95 in pixel unit, slice-wise
    if not _has_scipy:
        return float("nan")

    pred = (pred2d > 0).astype(np.uint8)
    gt = (gt2d > 0).astype(np.uint8)

    def _dt(a, b):
        dt = distance_transform_edt(1 - b)
        vals = dt[a.astype(bool)]
        if vals.size == 0:
            return np.array([0.0])
        return vals

    d1 = _dt(pred, gt)
    d2 = _dt(gt, pred)
    return float(np.percentile(np.hstack([d1, d2]), 95))


def boundary_error_heatmap(pred2d: np.ndarray, gt2d: np.ndarray) -> np.ndarray:
    # Boundary error heatmap via distance transform on contours
    if not _has_scipy:
        return None

    pc = V.mask_to_contour(pred2d.astype(np.uint8), thickness=1)
    gc = V.mask_to_contour(gt2d.astype(np.uint8), thickness=1)

    dt_to_gc = distance_transform_edt(1 - gc)
    dt_to_pc = distance_transform_edt(1 - pc)

    hm = np.zeros_like(pc, dtype=np.float32)
    hm = np.maximum(hm, dt_to_gc * pc.astype(np.float32))
    hm = np.maximum(hm, dt_to_pc * gc.astype(np.float32))
    return hm


@torch.no_grad()
def eval_hd95_on_test(data_root: str, weights_path: str, test_ids, base=32, pred_thr=0.5) -> float:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = V.UNetMini(base=base, p_drop=0.0).to(device)
    state = torch.load(weights_path, map_location=device)
    net.load_state_dict(state, strict=False)
    net.eval()

    images_dir = os.path.join(data_root, "imagesTr")
    labels_dir = os.path.join(data_root, "labelsTr")

    hd_list = []

    for cid in test_ids:
        img_path, lab_path = V.find_case_paths(data_root, "Tr", cid)
        if lab_path is None:
            continue

        img = V.load_nii(img_path).astype(np.float32)
        if img.ndim == 4:
            img = img[..., 0]
        gt = V.load_nii(lab_path)

        img_z = V.zscore_volume(img)
        pred_256 = V.predict_volume_slicewise(net, img_z, thr=pred_thr, device=device)
        gt_256 = V.build_gt_256(gt)

        # slice-wise HD95 mean (only slices with enough GT)
        for z in range(gt_256.shape[2]):
            if gt_256[:, :, z].sum() < 10:
                continue
            hd = hd95_2d(pred_256[:, :, z], gt_256[:, :, z])
            hd_list.append(hd)

    return float(np.nanmean(hd_list)) if len(hd_list) > 0 else float("nan")


def export_demo_heatmaps(data_root: str, weights_dict: dict, demo_case: str, outdir: str,
                         base=32, pred_thr=0.5):
    os.makedirs(outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img_path, lab_path = V.find_case_paths(data_root, "Tr", demo_case)
    if lab_path is None:
        raise RuntimeError("Demo case must have GT label (use imagesTr + labelsTr).")

    img = V.load_nii(img_path).astype(np.float32)
    if img.ndim == 4:
        img = img[..., 0]
    gt = V.load_nii(lab_path)

    img_z = V.zscore_volume(img)
    gt_256 = V.build_gt_256(gt)

    # pick slice with max GT area
    z = int(np.argmax([gt_256[:, :, k].sum() for k in range(gt_256.shape[2])]))

    # background for display
    bg2d = V.load_nii(img_path).astype(np.float32)
    if bg2d.ndim == 4:
        bg2d = bg2d[..., 0]
    bg2d = V.zscore_volume(bg2d)[:, :, z]
    bg2d = np.asarray(bg2d, dtype=np.float32)
    bg2d = cv2_resize_256(bg2d)
    vmin, vmax = np.percentile(bg2d, 1), np.percentile(bg2d, 99)
    bg = np.clip((bg2d - vmin) / (vmax - vmin + 1e-6), 0, 1)

    gt2d = gt_256[:, :, z].astype(np.uint8)

    for tag, wpath in weights_dict.items():
        net = V.UNetMini(base=base, p_drop=0.0).to(device)
        state = torch.load(wpath, map_location=device)
        net.load_state_dict(state, strict=False)
        net.eval()

        pred_256 = V.predict_volume_slicewise(net, img_z, thr=pred_thr, device=device)
        pred2d = pred_256[:, :, z].astype(np.uint8)

        hm = boundary_error_heatmap(pred2d, gt2d)
        if hm is None:
            print("[WARN] scipy not found, skip heatmap.")
            return

        plt.figure(figsize=(6, 6))
        plt.imshow(bg, cmap="gray")
        plt.imshow(hm, alpha=0.65)  # default colormap
        plt.axis("off")
        out_png = os.path.join(outdir, f"exp2_boundary_error_heatmap_{tag}_{demo_case}_z{z}.png")
        plt.tight_layout()
        plt.savefig(out_png, dpi=220, bbox_inches="tight", pad_inches=0)
        plt.close()
        print("[SAVE]", os.path.abspath(out_png))

    print("[INFO] Demo case:", demo_case, "slice z:", z)


def cv2_resize_256(x2d: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.resize(x2d.astype(np.float32), (256, 256)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to Task05_Prostate")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--dropout", required=True)
    ap.add_argument("--l2", required=True)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--pred_thr", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default=os.path.join("data", "reports", "exp2"))
    args = ap.parse_args()

    # case list from labelsTr (GT exists)
    labels_dir = os.path.join(args.data, "labelsTr")
    case_ids = sorted([os.path.basename(p).replace(".nii.gz", "")
                       for p in glob_nii(labels_dir)])
    _, _, test_ids = split_cases(case_ids, seed=args.seed)
    demo_case = test_ids[0] if len(test_ids) > 0 else case_ids[0]

    os.makedirs(args.outdir, exist_ok=True)

    hd_base = eval_hd95_on_test(args.data, args.baseline, test_ids, base=args.base, pred_thr=args.pred_thr)
    hd_drop = eval_hd95_on_test(args.data, args.dropout, test_ids, base=args.base, pred_thr=args.pred_thr)
    hd_l2   = eval_hd95_on_test(args.data, args.l2, test_ids, base=args.base, pred_thr=args.pred_thr)

    # save summary
    csv_path = os.path.join(args.outdir, "exp2_hd95_summary.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("model,hd95_mean_pix\n")
        f.write(f"baseline,{hd_base}\n")
        f.write(f"dropout,{hd_drop}\n")
        f.write(f"l2,{hd_l2}\n")
    print("[SAVE]", os.path.abspath(csv_path))

    # plot trend
    plt.figure(figsize=(7, 4))
    plt.plot(["baseline", "dropout", "l2"], [hd_base, hd_drop, hd_l2], marker="o")
    plt.title("Test HD95 (lower is better)")
    plt.xlabel("Regularization Strategy")
    plt.ylabel("HD95 (pixel)")
    plt.tight_layout()
    out_png = os.path.join(args.outdir, "exp2_hd95_trend.png")
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("[SAVE]", os.path.abspath(out_png))

    # heatmaps
    weights_dict = {"baseline": args.baseline, "dropout": args.dropout, "l2": args.l2}
    export_demo_heatmaps(args.data, weights_dict, demo_case=demo_case, outdir=args.outdir,
                         base=args.base, pred_thr=args.pred_thr)


def glob_nii(folder: str):
    import glob
    return glob.glob(os.path.join(folder, "*.nii.gz"))


if __name__ == "__main__":
    main()
