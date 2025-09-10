import argparse, os, glob
import numpy as np
import cv2


def imread_gray(path):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return im

def imread_color(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def load_prob(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".npy", ".npz"]:
        arr = np.load(path)
        if isinstance(arr, np.lib.npyio.NpzFile):
            for k in ["prob","probs","p","pred","out"]:
                if k in arr:
                    arr = arr[k]
                    break
            else:
                arr = arr[list(arr.keys())[0]]
        arr = arr.astype(np.float32)
    else:
        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise RuntimeError(f"Failed to read prob image: {path}")
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        arr = arr.astype(np.float32)
        mx = float(arr.max())
        if mx > 1.0:
            arr = arr / 255.0 if mx > 0 else np.zeros_like(arr, dtype=np.float32)
    return np.clip(arr, 0.0, 1.0)


def find_pairs(img_dir, msk_dir, prob_dir):
    def stem(p): return os.path.splitext(os.path.basename(p))[0]
    def norm_stem(s):
        s0 = s.lower()
        for suf in ["_mask","-mask","_gt","-gt","_label","-label","_labels","-labels",".mask",".gt",".label"]:
            if s0.endswith(suf):
                return s0[: -len(suf)]
        return s0

    imgs = {}
    for p in glob.glob(os.path.join(img_dir, "*")):
        if os.path.isfile(p):
            imgs[stem(p)] = p

    msks = {}
    for p in glob.glob(os.path.join(msk_dir, "*")):
        if os.path.isfile(p):
            s = stem(p)
            msks[s] = p
            msks[norm_stem(s)] = p

    prob_map = {}
    prob_files = [p for p in glob.glob(os.path.join(prob_dir, "**", "*"), recursive=True) if os.path.isfile(p)]
    for p in prob_files:
        s = stem(p)
        keys = {s, s.lower(), norm_stem(s)}
        for sep in ["_","-","."]:
            for tag in ["prob","probs","pred","out","p"]:
                if s.lower().endswith(f"{sep}{tag}"):
                    base = s[: -(len(tag)+1)]
                    keys.update({base, norm_stem(base)})
        for k in keys:
            prob_map.setdefault(k, p)

    pairs = []
    for s_img, ip in imgs.items():
        cand = None
        for k in [s_img, s_img.lower(), norm_stem(s_img)]:
            if k in msks and k in prob_map:
                cand = (ip, msks[k], prob_map[k])
                break
        if cand:
            pairs.append(cand)
    return pairs, len(imgs), len(msks), len(prob_files)


def dice_iou(pred_bin, gt_bin, eps=1e-6):
    pred = (pred_bin > 0).astype(np.float32)
    gt   = (gt_bin   > 0).astype(np.float32)
    inter = float((pred * gt).sum())
    a = float(pred.sum())
    b = float(gt.sum())
    dice = (2*inter + eps) / (a + b + eps)
    union = a + b - inter
    iou = (inter + eps) / (union + eps)
    return dice, iou


def put_title(img, text):
    out = img.copy()
    H, W = out.shape[:2]
    cv2.rectangle(out, (0,0), (W, 32), (0,0,0), -1)
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    return out

def panel_mask(gray_mask, H, W, title):
    m = gray_mask
    if m.shape != (H, W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    m3 = np.repeat((m > 0).astype(np.uint8)[...,None]*255, 3, axis=2)
    return put_title(m3, title)

def confusion_overlay(base_rgb, gt_mask, pred_mask, alpha=0.6, dilate_ks=3, dilate_iter=1):
    """TP=green, FP=red, FN=cyan. TN transparent. Optional dilation for visibility."""
    H, W = base_rgb.shape[:2]
    gt = gt_mask
    pr = pred_mask
    if gt.shape != (H,W):
        gt = cv2.resize(gt, (W,H), interpolation=cv2.INTER_NEAREST)
    if pr.shape != (H,W):
        pr = cv2.resize(pr, (W,H), interpolation=cv2.INTER_NEAREST)
    gt = (gt > 0).astype(np.uint8)
    pr = (pr > 0).astype(np.uint8)

    tp = (gt & pr) == 1
    fp = ((1 - gt) & pr) == 1
    fn = (gt & (1 - pr)) == 1


    if dilate_ks and dilate_iter:
        ksize = int(dilate_ks)
        if ksize % 2 == 0:  # ensure odd
            ksize += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        it = int(dilate_iter)
        tp = cv2.dilate(tp.astype(np.uint8), kernel, iterations=it).astype(bool)
        fp = cv2.dilate(fp.astype(np.uint8), kernel, iterations=it).astype(bool)
        fn = cv2.dilate(fn.astype(np.uint8), kernel, iterations=it).astype(bool)

    overlay = np.zeros_like(base_rgb, dtype=np.uint8)
    overlay[tp] = (0,255,0)        # green
    overlay[fp] = (255,0,0)        # red
    overlay[fn] = (0,200,255)      # cyan

    blended = cv2.addWeighted(base_rgb, 1.0, overlay, float(alpha), 0.0)

    # legend banner
    cv2.rectangle(blended, (0,0), (W, 36), (0,0,0), -1)
    cv2.putText(blended, "Overlay  TP=green  FP=red  FN=cyan",
                (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    # color swatches
    cv2.rectangle(blended, (8, 40), (28, 60), (0,255,0), -1);      cv2.putText(blended, "TP", (34,58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    cv2.rectangle(blended, (80, 40), (100, 60), (255,0,0), -1);    cv2.putText(blended, "FP", (106,58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    cv2.rectangle(blended, (152, 40), (172, 60), (0,200,255), -1); cv2.putText(blended, "FN", (178,58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    return blended

def make_grid_three(img_rgb, gt_mask, prob, thr, alpha, viz_dilate_ks=3, viz_dilate_iter=1):
    H, W = img_rgb.shape[:2]
    base = img_rgb if img_rgb.ndim == 3 else cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)

    # resize prob & GT to image size
    if prob.shape != (H, W):
        prob = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)
    if gt_mask.shape != (H, W):
        gt_mask = cv2.resize(gt_mask, (W, H), interpolation=cv2.INTER_NEAREST)

    # binarize
    pred_bin = (prob >= float(thr)).astype(np.uint8) * 255
    gt_bin   = (gt_mask > 0).astype(np.uint8) * 255

    # panels
    p0 = put_title(base, "Original")
    p1 = panel_mask(gt_bin, H, W, "GT mask")
    p2 = confusion_overlay(base, gt_bin, pred_bin, alpha=alpha,
                           dilate_ks=viz_dilate_ks, dilate_iter=viz_dilate_iter)

    # metrics
    d, i = dice_iou(pred_bin, gt_bin)
    cv2.putText(p2, f"Dice={d:.4f}  IoU={i:.4f}", (8, H-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

    # 3-wide grid
    grid = np.concatenate([p0, p1, p2], axis=1)
    return grid, pred_bin



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--masks_dir",  required=True)
    ap.add_argument("--probs_dir",  required=True, help="Can point to parent; search is recursive.")
    ap.add_argument("--out_dir",    required=True)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--positives_only", action="store_true")
    # display-only dilation controls
    ap.add_argument("--viz_dilate", type=int, default=3, help="Odd kernel size for visualization dilation (0 disables).")
    ap.add_argument("--viz_dilate_iter", type=int, default=1, help="Dilation iterations for visualization.")
    args = ap.parse_args()

    pairs, n_img, n_msk, n_prob = find_pairs(args.images_dir, args.masks_dir, args.probs_dir)

    if args.positives_only:
        filtered = []
        for ip, mp, pp in pairs:
            m = imread_gray(mp)
            if (m > 0).any():
                filtered.append((ip, mp, pp))
        pairs = filtered

    if args.limit > 0:
        pairs = pairs[:args.limit]

    if len(pairs) == 0:
        print(f"[visualize] Found 0 pairs. (images={n_img}, masks={n_msk}, probs={n_prob}). "
              f"Point --probs_dir to the folder that contains probs, e.g. '.../probs_png' or '.../probs_npy'.")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    saved = 0

    for ip, mp, pp in pairs:
        img = imread_color(ip)
        mask = imread_gray(mp)
        prob = load_prob(pp)

        stem = os.path.splitext(os.path.basename(ip))[0]
        grid, pred_bin = make_grid_three(
            img, mask, prob, args.thr, args.alpha,
            viz_dilate_ks=args.viz_dilate, viz_dilate_iter=args.viz_dilate_iter
        )

        out_grid = os.path.join(args.out_dir, f"{stem}_grid3.png")
        out_pred = os.path.join(args.out_dir, f"{stem}_pred_thr{args.thr:.2f}.png")
        cv2.imwrite(out_grid, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        cv2.imwrite(out_pred, pred_bin)
        saved += 1

    print(f"Saved {saved} grids to: {args.out_dir} "
          f"(matched out of images={n_img}, masks={n_msk}, probs={n_prob})")

if __name__ == "__main__":
    main()
