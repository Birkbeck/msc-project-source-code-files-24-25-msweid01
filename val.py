# val.py
import os, argparse, yaml, numpy as np, torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import make_loaders
from archs import HybridEffB3ViTUNet
from metrics import dice_coef, iou_score, roc_auc, froc_metrics

def save_png_mask(arr01, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import cv2
    cv2.imwrite(path, (arr01 * 255).astype(np.uint8))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="runs/eval")
    ap.add_argument("--thr", type=float, default=0.5)

    # detection params / scan range
    ap.add_argument("--merge_radius", type=int, default=3)
    ap.add_argument("--min_area", type=int, default=9)
    ap.add_argument("--scan_min", type=float, default=0.10)
    ap.add_argument("--scan_max", type=float, default=0.90)
    ap.add_argument("--scan_steps", type=int, default=17)

    # NEW: morphology / shape filters for FROC
    ap.add_argument("--open_k", type=int, default=0)
    ap.add_argument("--close_k", type=int, default=0)
    ap.add_argument("--min_circ", type=float, default=0.0)
    ap.add_argument("--min_sol", type=float, default=0.0)
    ap.add_argument("--filter_border", action="store_true")
    ap.add_argument("--border_pad", type=int, default=0)

    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--save_probs", action="store_true")
    ap.add_argument("--save_png", action="store_true")
    ap.add_argument("--save_preds", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get only the val loader
    _, val_loader = make_loaders(args.data_root, args.fold,
                                 batch_size=1, num_workers=args.num_workers,
                                 balance=False, imagenet_norm=False)

    # build model and load checkpoint (non-strict to allow minor arch diffs)
    model = HybridEffB3ViTUNet(num_classes=1).to(device)
    sd = torch.load(args.ckpt, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    incomp = model.load_state_dict(sd, strict=False)
    if incomp.missing_keys or incomp.unexpected_keys:
        print("== Checkpoint loaded with non-strict matching ==")
        if incomp.missing_keys:
            print("Missing keys:", list(incomp.missing_keys))
        if incomp.unexpected_keys:
            print("Unexpected keys:", list(incomp.unexpected_keys))

    model.eval()

    # output dirs
    probs_dir = os.path.join(args.out, "probs_npy")
    png_dir   = os.path.join(args.out, "probs_png")
    preds_dir = os.path.join(args.out, f"preds_thr_{int(round(args.thr*100))}")
    if args.save_probs: os.makedirs(probs_dir, exist_ok=True)
    if args.save_png:   os.makedirs(png_dir, exist_ok=True)
    if args.save_preds: os.makedirs(preds_dir, exist_ok=True)

    preds = []   # list of HxW floats [0..1]
    gts   = []   # list of HxW {0,1}
    names = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Infer")
        for bi, (x, y) in enumerate(pbar):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            out = model(x)                # Bx1xHxW probs
            p  = out[0,0].detach().cpu().numpy().astype(np.float32)
            g  = y[0,0].detach().cpu().numpy().astype(np.uint8)

            # derive name from dataset file list
            stem = os.path.splitext(os.path.basename(val_loader.dataset.img_paths[bi]))[0]
            names.append(stem)

            preds.append(p)
            gts.append(g)

            if args.save_probs:
                np.save(os.path.join(probs_dir, stem + ".npy"), p)
            if args.save_png:
                save_png_mask(np.clip(p,0,1), os.path.join(png_dir, stem + ".png"))
            if args.save_preds:
                pb = (p >= args.thr).astype(np.uint8)
                save_png_mask(pb, os.path.join(preds_dir, stem + ".png"))

    # --- auto-threshold by Dice on positives ---
    thrs = np.linspace(args.scan_min, args.scan_max, args.scan_steps)
    best_thr = None
    best_dice_pos = -1.0
    for t in thrs:
        dices_pos = []
        for p, g in zip(preds, gts):
            if (g > 0).any():
                dices_pos.append(dice_coef(p, g, thr=t))
        if len(dices_pos):
            m = float(np.nanmean(dices_pos))
            if m > best_dice_pos:
                best_dice_pos = m
                best_thr = float(t)
    if best_thr is None:
        best_thr = args.thr
        best_dice_pos = 0.0

    print(f"[auto] best_thr by Dice on positive images = {best_thr:.3f} (mean Dice={best_dice_pos:.4f})")

    # --- metrics @ auto thr ---
    def eval_at_thr(t):
        d_all, j_all, a_all = [], [], []
        d_pos, j_pos, a_pos = [], [], []
        for p, g in zip(preds, gts):
            d = dice_coef(p, g, thr=t)
            j = iou_score(p, g, thr=t)
            a = roc_auc(p, g)
            d_all.append(d); j_all.append(j); a_all.append(a)
            if (g > 0).any():
                d_pos.append(d); j_pos.append(j); a_pos.append(a)
        def m(x): return float(np.nanmean(x)) if len(x) else float("nan")
        return {
            "thr": float(t),
            "overall": {"dice_mean": m(d_all), "dice_std": float(np.nanstd(d_all)),
                        "iou_mean": m(j_all),  "iou_std": float(np.nanstd(j_all)),
                        "auc_mean": m(a_all),  "auc_std": float(np.nanstd(a_all))},
            "positives_only": {"count": int(sum(1 for g in gts if (g>0).any())),
                               "dice_mean": m(d_pos), "dice_std": float(np.nanstd(d_pos)),
                               "iou_mean": m(j_pos),  "iou_std": float(np.nanstd(j_pos)),
                               "auc_mean": m(a_pos),  "auc_std": float(np.nanstd(a_pos))}
        }

    metrics_auto = eval_at_thr(best_thr)
    metrics_ref  = eval_at_thr(args.thr)

    # --- FROC with morphology / shape filters ---
    froc = froc_metrics(
        preds, gts,
        thresholds=thrs.tolist(),
        fppi_targets=[0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
        merge_radius=args.merge_radius, min_area=args.min_area,
        open_kernel=args.open_k, close_kernel=args.close_k,
        min_circularity=args.min_circ, min_solidity=args.min_sol,
        filter_border=args.filter_border, border_pad=args.border_pad
    )

    summary = {
        "auto_best_thr": float(best_thr),
        "metrics_at_auto_thr": metrics_auto,
        "metrics_at_ref_thr": metrics_ref,
        "froc": froc,
    }
    with open(os.path.join(args.out, "summary.yaml"), "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    print("== Summary ==")
    print(yaml.safe_dump(summary, sort_keys=False))

if __name__ == "__main__":
    main()
