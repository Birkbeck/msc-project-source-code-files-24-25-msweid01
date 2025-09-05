import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from skimage.measure import label, regionprops
from scipy.spatial.distance import cdist
import cv2
import math


def _to_bool(pred, gt, thr=0.5):
    p = (pred >= thr)
    g = (gt > 0)
    return np.ascontiguousarray(p.astype(np.bool_)), np.ascontiguousarray(g.astype(np.bool_))

def dice_coef(pred, gt, thr=0.5, eps=1e-7):
    p, g = _to_bool(pred, gt, thr)
    tp = np.logical_and(p, g).sum()
    fp = np.logical_and(p, ~g).sum()
    fn = np.logical_and(~p, g).sum()
    return (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)

def iou_score(pred, gt, thr=0.5, eps=1e-7):
    p, g = _to_bool(pred, gt, thr)
    tp = np.logical_and(p, g).sum()
    fp = np.logical_and(p, ~g).sum()
    fn = np.logical_and(~p, g).sum()
    return (tp + eps) / (tp + fp + fn + eps)

def roc_auc(pred, gt):
    y_true = (gt.flatten() > 0).astype(np.uint8)
    y_score = pred.flatten().astype(np.float32)
    if y_true.max() == y_true.min():
        return float("nan")
    return roc_auc_score(y_true, y_score)

def average_precision(pred, gt):
    y_true = (gt.flatten() > 0).astype(np.uint8)
    y_score = pred.flatten().astype(np.float32)
    if y_true.max() == y_true.min():
        return float("nan")
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    return auc(rec, prec)

# ---------- helpers for FROC ----------
def _circularity(area, perimeter, eps=1e-6):
    # 4πA / P^2 in [0,1] for ideal circles
    return float(4.0 * math.pi * area / ((perimeter + eps) ** 2))

def _touches_border(bbox, H, W, pad=0):
    y0, x0, y1, x1 = bbox  # (min_row, min_col, max_row, max_col)
    return (y0 <= pad) or (x0 <= pad) or (y1 >= H - pad) or (x1 >= W - pad)

def preprocess_mask(mask_uint8,
                    open_kernel=0, close_kernel=0,
                    min_area=9, max_area=None,
                    min_circularity=0.0, min_solidity=0.0,
                    filter_border=False, border_pad=0):

    m = (mask_uint8 > 0).astype(np.uint8) * 255

    if open_kernel and open_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    if close_kernel and close_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

    lab = label(m > 0, connectivity=2)
    H, W = m.shape
    comps = []
    for r in regionprops(lab):
        if min_area and r.area < min_area:
            continue
        if max_area and r.area > max_area:
            continue
        if min_solidity and r.solidity < min_solidity:
            continue
        peri = r.perimeter if r.perimeter is not None else 0.0
        if min_circularity and _circularity(r.area, peri) < min_circularity:
            continue
        if filter_border and _touches_border(r.bbox, H, W, pad=border_pad):
            continue
        comps.append(r)
    return comps

def _centroids(comps):
    if not comps:
        return np.zeros((0,2), dtype=np.float32)
    return np.array([c.centroid for c in comps], dtype=np.float32)

def froc_metrics(pred_probs_list, gt_masks_list, thresholds=None, fppi_targets=None,
                 merge_radius=3, min_area=9,
                 open_kernel=0, close_kernel=0, min_circularity=0.0, min_solidity=0.0,
                 filter_border=False, border_pad=0):

    if thresholds is None:
        thresholds = np.linspace(0.10, 0.90, 17)
    if fppi_targets is None:
        fppi_targets = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

    num_images = len(pred_probs_list)
    assert num_images == len(gt_masks_list)

    # Prepare GT centroids
    gt_centroids = []
    total_gt = 0
    for gt in gt_masks_list:
        g = (gt > 0).astype(np.uint8) * 255
        comps = preprocess_mask(g, open_kernel=0, close_kernel=0,
                                min_area=min_area, max_area=None,
                                min_circularity=0.0, min_solidity=0.0,
                                filter_border=False, border_pad=0)
        c = _centroids(comps)
        gt_centroids.append(c)
        total_gt += len(c)

    ops = []
    for thr in thresholds:
        TP = 0
        FP = 0
        for pred, gcent in zip(pred_probs_list, gt_centroids):
            P = (pred >= thr).astype(np.uint8) * 255
            pcomps = preprocess_mask(
                P, open_kernel=open_kernel, close_kernel=close_kernel,
                min_area=min_area, max_area=None,
                min_circularity=min_circularity, min_solidity=min_solidity,
                filter_border=filter_border, border_pad=border_pad
            )
            pc = _centroids(pcomps)

            if len(pc) == 0:
                continue
            if len(gcent) == 0:
                FP += len(pc)
                continue

            D = cdist(pc, gcent)
            tp_mask = (D.min(axis=1) <= merge_radius)
            TP += int(tp_mask.sum())
            FP += int((~tp_mask).sum())

        FN = total_gt - TP
        fppi = FP / max(num_images, 1)
        sens = TP / max(total_gt, 1e-8)
        ops.append({"thr": float(thr), "TP": int(TP), "FP": int(FP), "FN": int(FN),
                    "fppi": float(fppi), "sensitivity": float(sens)})

    # Select operating points: best sensitivity with fppi <= target; if none, pick min fppi above target
    sens_at = {}
    closest = {}
    for tgt in fppi_targets:
        below = [o for o in ops if o["fppi"] <= tgt + 1e-9]
        if len(below):
            best = max(below, key=lambda o: o["sensitivity"])
        else:
            best = min(ops, key=lambda o: o["fppi"])
        sens_at[str(tgt)] = best["sensitivity"]
        closest[str(tgt)] = {k: best[k] for k in ["thr", "TP", "FP", "FN", "fppi"]}
        closest[str(tgt)]["sensitivity"] = best["sensitivity"]

    cpm = float(np.mean(list(sens_at.values())))

    ap = float(average_precision(
        np.concatenate([p.flatten() for p in pred_probs_list]),
        np.concatenate([g.flatten() for g in gt_masks_list])
    ))

    return {
        "fppi_targets": fppi_targets,
        "sensitivity_at_fppi": sens_at,
        "cpm": cpm,
        "operating_points_closest": closest,
        "num_images": num_images,
        "total_gt_lesions": int(total_gt),
        "ap": ap,
    }
