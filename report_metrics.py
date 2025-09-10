import os, argparse, glob, math, csv
import numpy as np
import cv2
from tqdm import tqdm
import yaml
import matplotlib.pyplot as plt
from typing import List, Dict


def _parse_thr_list(s):
    if not s:
        return None
    parts = [p for chunk in s.split(',') for p in chunk.split()]
    vals = []
    for p in parts:
        try:
            vals.append(float(p))
        except ValueError:
            pass
    vals = [v for v in vals if 0.0 <= v <= 1.0]
    return sorted(set(vals)) if vals else None

def _parse_fppi_points(s):
    if not s or str(s).strip() == "":
        return [0.125, 0.25, 0.5, 1, 2, 4, 8]
    parts = [p for chunk in str(s).split(',') for p in chunk.split()]
    vals = []
    for p in parts:
        try:
            vals.append(float(p))
        except ValueError:
            pass
    return [v for v in vals if v >= 0]

def _parse_size_bins(s: str) -> List[int]:
    if not s:
        return [5, 10]
    parts = [p for chunk in str(s).split(',') for p in chunk.split()]
    vals = []
    for p in parts:
        try:
            v = int(p)
            if v > 0:
                vals.append(v)
        except ValueError:
            pass
    vals = sorted(set(vals))
    return vals if vals else [5, 10]


def read_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img

def binarize_from_probs(prob_img, thr):
    arr = prob_img.astype(np.float32)
    if arr.max() > 1.0:
        arr /= 65535.0 if arr.max() > 255 else 255.0
    return (arr >= float(thr)).astype(np.uint8)


def connected_components(mask):
    return cv2.connectedComponentsWithStats(mask, connectivity=8)

def apply_min_area(bin_img, min_area):
    if min_area <= 1:
        return bin_img
    num, labels, stats, _ = connected_components(bin_img)
    out = np.zeros_like(bin_img, dtype=np.uint8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 1
    return out

def merge_close_components(bin_img, radius: int):
    r = int(radius)
    if r <= 0:
        return bin_img
    k = 2 * r + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    merged = cv2.morphologyEx(bin_img.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return (merged > 0).astype(np.uint8)

def _centroid_distance(c1, c2):
    dy = c1[1] - c2[1]
    dx = c1[0] - c2[0]
    return math.hypot(dx, dy)

def _compute_match(
    gt_lab, gt_stats, gt_centroids,
    pr_lab, pr_stats, pr_centroids,
    mode: str = "overlap",
    centroid_radius: int = 3,
    iou_thr: float = 0.0
):

    H, W = gt_lab.shape
    num_g = int(gt_stats.shape[0])
    num_p = int(pr_stats.shape[0])

    gt_ids = [i for i in range(1, num_g)]
    pr_ids = [i for i in range(1, num_p)]

    gt_hit = {gid: False for gid in gt_ids}
    tp = 0
    fp = 0

    if mode == "centroid":
        gt_bin = (gt_lab > 0).astype(np.uint8)

    for pid in pr_ids:
        pred_mask = (pr_lab == pid)
        is_tp = False
        hit_gids = []

        if mode == "overlap":
            intersect_ids = np.unique(gt_lab[pred_mask])
            hit_gids = [gid for gid in intersect_ids.tolist() if gid != 0]
            is_tp = len(hit_gids) > 0

        elif mode == "centroid":
            cx, cy = pr_centroids[pid]
            x0 = max(0, int(round(cx - centroid_radius)))
            x1 = min(W, int(round(cx + centroid_radius + 1)))
            y0 = max(0, int(round(cy - centroid_radius)))
            y1 = min(H, int(round(cy + centroid_radius + 1)))
            patch = gt_lab[y0:y1, x0:x1]
            if patch.size > 0 and (patch > 0).any():
                hit_gids = np.unique(patch[patch > 0]).tolist()
                is_tp = True

        elif mode == "iou":
            pred_area = int(pr_stats[pid, cv2.CC_STAT_AREA])
            if pred_area == 0:
                is_tp = False
            else:
                px, py, pw, ph, _ = pr_stats[pid]
                roi = gt_lab[py:py+ph, px:px+pw]
                roi_pred = (pr_lab[py:py+ph, px:px+pw] == pid)
                cand_gids = np.unique(roi[roi > 0]).tolist()
                best_iou = 0.0
                for gid in cand_gids:
                    gt_area = int(gt_stats[gid, cv2.CC_STAT_AREA])
                    inter = int((roi_pred & (roi == gid)).sum())
                    if inter == 0:
                        continue
                    union = pred_area + gt_area - inter
                    iou = inter / max(1, union)
                    if iou >= iou_thr:
                        hit_gids.append(gid)
                        best_iou = max(best_iou, iou)
                is_tp = best_iou >= iou_thr

        else:
            raise ValueError(f"Unknown match mode: {mode}")

        if is_tp:
            tp += 1
            for gid in hit_gids:
                gt_hit[gid] = True
        else:
            fp += 1

    return gt_hit, tp, fp

def evaluate_case(gt_mask, pred_bin, match_mode="overlap", centroid_radius=3, iou_thr=0.0):
    num_g, lab_g, stats_g, cents_g = connected_components(gt_mask.astype(np.uint8))
    num_p, lab_p, stats_p, cents_p = connected_components(pred_bin.astype(np.uint8))

    gt_hit, tp, fp = _compute_match(
        lab_g, stats_g, cents_g, lab_p, stats_p, cents_p,
        mode=match_mode, centroid_radius=centroid_radius, iou_thr=iou_thr
    )
    fn = sum(1 for gid in range(1, num_g) if not gt_hit.get(gid, False))
    gt_areas = [int(stats_g[i, cv2.CC_STAT_AREA]) for i in range(1, num_g)]
    return tp, fp, fn, len(gt_areas), gt_areas


def compute_froc(points, num_images, fppi_targets, tp_mode='pred'):
    records = []
    for p in points:
        fppi = p['FP'] / max(1, num_images)
        if tp_mode == 'gt':
            sens = (p['N_gt'] - p['FN']) / max(1, p['N_gt'])
        else:
            sens = p['TP'] / max(1, p['N_gt'])
        records.append((fppi, sens, p['thr'], p))
    records.sort(key=lambda x: x[0])
    xs = [r[0] for r in records]
    ys = [r[1] for r in records]
    def interp(xq):
        if not xs: return 0.0
        if xq <= xs[0]: return ys[0]
        if xq >= xs[-1]: return ys[-1]
        for i in range(1, len(xs)):
            if xs[i] >= xq:
                x0,y0,x1,y1 = xs[i-1],ys[i-1],xs[i],ys[i]
                t = 0.0 if x1==x0 else (xq-x0)/(x1-x0)
                return y0 + t*(y1-y0)
        return ys[-1]
    sens_at = {fp: float(interp(fp)) for fp in fppi_targets}
    cpm = float(np.mean(list(sens_at.values()))) if sens_at else float('nan')
    return sens_at, cpm, records


def compute_precision_recall(points):
    pts = sorted(points, key=lambda d: d['thr'])
    prec, rec, thr = [], [], []
    for p in pts:
        tp, fp, ngt = p['TP'], p['FP'], p['N_gt']
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, ngt)
        prec.append(precision)
        rec.append(recall)
        thr.append(p['thr'])

    order = np.argsort(rec)
    rec_sorted = np.array(rec)[order]
    prec_sorted = np.array(prec)[order]
    rec_sorted, prec_sorted = rec_sorted.tolist(), prec_sorted.tolist()
    for i in range(len(prec_sorted)-2, -1, -1):
        prec_sorted[i] = max(prec_sorted[i], prec_sorted[i+1])

    trap = getattr(np, "trapezoid", np.trapz)
    ap = float(trap(prec_sorted, rec_sorted)) if len(rec_sorted) > 1 else float('nan')

    # Best F1 across thresholds
    f1s = []
    for pr, rc in zip(prec, rec):
        f1 = 0.0 if (pr + rc) == 0 else 2 * pr * rc / (pr + rc)
        f1s.append(f1)
    best_idx = int(np.argmax(f1s)) if f1s else 0
    best_f1 = float(f1s[best_idx]) if f1s else 0.0
    best_thr = float(thr[best_idx]) if thr else float('nan')

    return {
        'thresholds': [float(t) for t in thr],
        'precision': [float(p) for p in prec],
        'recall': [float(r) for r in rec],
        'ap': ap,
        'best_f1': best_f1,
        'best_f1_thr': best_thr,
        'precision_envelope': [float(p) for p in prec_sorted],
        'recall_sorted': [float(r) for r in rec_sorted],
    }

def find_operating_points(points, num_images, fppi_targets):
    best = {}
    for fp_target in fppi_targets:
        best_rec = None
        best_diff = float('inf')
        for p in points:
            fppi = p['FP'] / max(1, num_images)
            diff = abs(fppi - fp_target)
            if diff < best_diff:
                best_diff = diff
                best_rec = (fppi, p)
        if best_rec is None:
            continue
        fppi, p = best_rec
        tp, fp, fn, ngt = p['TP'], p['FP'], p['FN'], p['N_gt']
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, ngt)
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        best[str(fp_target)] = {
            'thr': float(p['thr']),
            'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'N_gt': int(ngt),
            'fppi': float(fppi),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
        }
    return best

def size_binned_sensitivity(gt_areas_all: List[List[int]], gt_hit_all: List[List[bool]], size_bins: List[int]):
    if not gt_areas_all:
        return {}
    edges = size_bins[:]  # ascending
    bin_labels = []
    # build ranges
    lows = [0] + [e+1 for e in edges]
    highs = edges + [10**9]
    for lo, hi in zip(lows, highs):
        if hi == 10**9:
            bin_labels.append(f">={lo}")
        elif lo == 0:
            bin_labels.append(f"<={hi}")
        else:
            bin_labels.append(f"{lo}-{hi}")

    # aggregate
    hits = [0] * len(lows)
    totals = [0] * len(lows)
    for areas, hits_per_img in zip(gt_areas_all, gt_hit_all):
        for a, h in zip(areas, hits_per_img):
            for bi, (lo, hi) in enumerate(zip(lows, highs)):
                if (lo == 0 and a <= hi) or (lo > 0 and lo <= a <= hi):
                    totals[bi] += 1
                    hits[bi] += int(bool(h))
                    break
    sens = [ (hits[i] / totals[i]) if totals[i] > 0 else float('nan') for i in range(len(totals)) ]
    return {'bins': bin_labels, 'totals': totals, 'hits': hits, 'sensitivity': [float(s) for s in sens]}


def _bootstrap_cis(per_image_by_thr: Dict[float, List[Dict]], fppi_targets, num_images, iters=0, seed=12
    if iters <= 0:
        return {}

    rng = np.random.default_rng(seed)
    fppi_targets = sorted(set(fppi_targets))
    cpm_vals = []
    sens_matrix = {str(fp): [] for fp in fppi_targets}

    thrs = sorted(per_image_by_thr.keys())

    for _ in range(iters):
        # sample image indices with replacement
        idx = rng.integers(0, num_images, size=num_images)

        # aggregate totals per threshold for this resample
        points = []
        for thr in thrs:
            arr = per_image_by_thr[thr]
            tp = sum(arr[i]['TP'] for i in idx if i < len(arr))
            fp = sum(arr[i]['FP'] for i in idx if i < len(arr))
            fn = sum(arr[i]['FN'] for i in idx if i < len(arr))
            ng = sum(arr[i]['N_gt'] for i in idx if i < len(arr))
            points.append({'thr': thr, 'TP': tp, 'FP': fp, 'FN': fn, 'N_gt': ng})

        sens_at, cpm, _ = compute_froc(points, num_images, fppi_targets)
        cpm_vals.append(cpm)
        for k, v in sens_at.items():
            sens_matrix[str(k)].append(v)

    def _ci(arr):
        if not arr:
            return [float('nan'), float('nan')]
        lo, hi = np.percentile(arr, [2.5, 97.5])
        return [float(lo), float(hi)]

    out = {
        'bootstrap': {
            'n': int(iters),
            'cpm_ci95': _ci(cpm_vals),
            'sens_at_fppi_ci95': {str(k): _ci(v) for k, v in sens_matrix.items()}
        }
    }
    return out

# ------------------------- Main -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img_dir', required=True, help='Directory of validation images (for ID reference only)')
    ap.add_argument('--gt_dir', required=True, help='Directory of GT masks (binary PNGs)')
    ap.add_argument('--prob_dir', required=True, help='Directory of predicted probability maps or binary masks')
    ap.add_argument('--thr', type=float, default=None, help='Single threshold on probability maps [0..1]')
    ap.add_argument('--thr_sweep', type=str, default=None, help='List of thresholds, e.g. "0.004,0.005,0.01"')
    ap.add_argument('--min_area', type=int, default=2, help='Remove predicted components smaller than this many pixels')
    ap.add_argument('--fppi_points', type=str, default=None, help='FP/image points, e.g. "0.125,0.25,0.5,1,2,4,8"')
    ap.add_argument('--max_fppi', type=float, default=8.0, help='Clamp FPPI axis to this maximum in the plot')
    ap.add_argument('--out', type=str, default='metrics.yaml', help='Output YAML path')
    ap.add_argument('--plot', action='store_true', help='Save FROC and PR plots next to YAML (PNGs)')
    ap.add_argument('--per_image_csv', type=str, default=None, help='Optional CSV of per-image TP/FP/FN')

    # NEW knobs
    ap.add_argument('--match_mode', type=str, default='overlap', choices=['overlap', 'centroid', 'iou'],
                    help="Object matching rule: 'overlap' (default), 'centroid', or 'iou'")
    ap.add_argument('--centroid_radius', type=int, default=3, help='Radius in pixels for centroid matching')
    ap.add_argument('--iou_thr', type=float, default=0.0, help='IoU threshold for IoU-based matching')

    ap.add_argument('--size_bins', type=str, default="5,10",
                    help='Comma list of pixel area edges for size-binned recall (default "5,10" -> <=5, 6-10, >=11)')
    ap.add_argument('--size_sens_at_fppi', type=float, default=1.0,
                    help='Pick the threshold closest to this FPPI to compute size-binned sensitivity')

    ap.add_argument('--bootstrap', type=int, default=0, help='Bootstrap iterations for 95%% CIs (0 to disable)')
    ap.add_argument('--bootstrap_seed', type=int, default=123, help='Bootstrap RNG seed')

    # ✅ New: merge radius for morphological de-duplication
    ap.add_argument('--merge_radius', type=int, default=0,
                    help='Morphological closing radius (px) to merge nearby predicted blobs (0=off)')
    ap.add_argument('--froc_tp_by', type=str, default='pred',
                    choices=['pred', 'gt'],
                    help='How to count TP for FROC/CPM: per prediction ("pred") or per GT hit ("gt"=N_gt-FN). PR/AP unchanged.')

    args = ap.parse_args()

    thr_list = _parse_thr_list(args.thr_sweep)
    if args.thr is not None:
        thr_list = sorted(set((thr_list or []) + [float(args.thr)])) if thr_list else [float(args.thr)]
    if not thr_list:
        thr_list = [0.0085]

    fppi_targets = _parse_fppi_points(args.fppi_points)
    size_bins = _parse_size_bins(args.size_bins)

    # Collect image IDs (drives consistent ordering & counts)
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(args.img_dir, '*.png'))]
    img_ids.sort()
    if not img_ids:
        raise RuntimeError(f"No images found in {args.img_dir}")

    # Map ID → paths
    gt_map = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(args.gt_dir, '*.png'))}
    pr_map = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(args.prob_dir, '*.png'))}

    # Per-image CSV rows
    per_rows = []

    # To support bootstrap: keep per-image results per threshold
    per_image_by_thr: Dict[float, List[Dict]] = {t: [] for t in thr_list}

    # Also keep per-image GT size arrays + hit flags for size-binned recall (for ONE selected thr later)
    size_gt_areas_per_img_at_thr = None
    size_gt_hits_per_img_at_thr = None

    results_by_thr = []
    for thr in thr_list:
        total_TP = total_FP = total_FN = total_gt = 0
        per_img_records = []

        for img_id in tqdm(img_ids, desc=f"Thr={thr:.6f}"):
            gt_path = gt_map.get(img_id, None)
            pr_path = pr_map.get(img_id, None)
            if gt_path is None or pr_path is None:
                # skip missing pairs silently
                continue

            gt = read_gray(gt_path)
            gt_bin = (gt > 127).astype(np.uint8)

            pred_raw = read_gray(pr_path)
            pred_bin = binarize_from_probs(pred_raw, thr)
            pred_bin = apply_min_area(pred_bin, args.min_area)
            pred_bin = merge_close_components(pred_bin, args.merge_radius)

            # Evaluate + capture GT areas and hits for optional size-binning later
            num_g, lab_g, stats_g, cents_g = connected_components(gt_bin)
            num_p, lab_p, stats_p, cents_p = connected_components(pred_bin)
            gt_hit, tp, fp = _compute_match(
                lab_g, stats_g, cents_g, lab_p, stats_p, cents_p,
                mode=args.match_mode, centroid_radius=args.centroid_radius, iou_thr=args.iou_thr
            )
            fn = sum(1 for gid in range(1, num_g) if not gt_hit.get(gid, False))
            gt_areas = [int(stats_g[i, cv2.CC_STAT_AREA]) for i in range(1, num_g)]
            gt_hits_list = [bool(gt_hit.get(i, False)) for i in range(1, num_g)]

            total_TP += tp
            total_FP += fp
            total_FN += fn
            total_gt += len(gt_areas)

            per_img_records.append({'img_id': img_id, 'TP': tp, 'FP': fp, 'FN': fn, 'N_gt': len(gt_areas)})

            if args.per_image_csv:
                per_rows.append((img_id, tp, fp, fn, len(gt_areas)))

        results_by_thr.append({
            'thr': float(thr),
            'TP': int(total_TP),
            'FP': int(total_FP),
            'FN': int(total_FN),
            'N_gt': int(total_gt),
            'num_images': int(len(img_ids)),
        })
        per_image_by_thr[thr] = per_img_records

    # FROC + CPM
    fppi_targets = sorted(set(fppi_targets))
    sens_at_fppi, froc_score, records = compute_froc(results_by_thr, len(img_ids), fppi_targets, tp_mode=args.froc_tp_by)

    # Exact operating points (closest thresholds to targets)
    operating_points = find_operating_points(results_by_thr, len(img_ids), fppi_targets)

    # Precision–Recall + AP + best F1
    pr_pkg = compute_precision_recall(results_by_thr)

    # Choose threshold nearest to requested FPPI for size-binned recall
    target = float(args.size_sens_at_fppi)
    closest_thr = None
    best_diff = float('inf')
    for p in results_by_thr:
        diff = abs((p['FP'] / max(1, len(img_ids))) - target)
        if diff < best_diff:
            best_diff = diff
            closest_thr = p['thr']

    # Recompute per-image GT areas + hits at that threshold to produce size-binned sensitivity
    size_bin_stats = {}
    if closest_thr is not None:
        size_gt_areas_per_img_at_thr = []
        size_gt_hits_per_img_at_thr = []

        for img_id in img_ids:
            gt_path = gt_map.get(img_id, None)
            pr_path = pr_map.get(img_id, None)
            if gt_path is None or pr_path is None:
                continue

            gt = read_gray(gt_path)
            gt_bin = (gt > 127).astype(np.uint8)

            pred_raw = read_gray(pr_path)
            pred_bin = binarize_from_probs(pred_raw, closest_thr)
            pred_bin = apply_min_area(pred_bin, args.min_area)
            pred_bin = merge_close_components(pred_bin, args.merge_radius)

            num_g, lab_g, stats_g, cents_g = connected_components(gt_bin)
            num_p, lab_p, stats_p, cents_p = connected_components(pred_bin)
            gt_hit, tp, fp = _compute_match(
                lab_g, stats_g, cents_g, lab_p, stats_p, cents_p,
                mode=args.match_mode, centroid_radius=args.centroid_radius, iou_thr=args.iou_thr
            )

            gt_areas = [int(stats_g[i, cv2.CC_STAT_AREA]) for i in range(1, num_g)]
            gt_hits_list = [bool(gt_hit.get(i, False)) for i in range(1, num_g)]

            size_gt_areas_per_img_at_thr.append(gt_areas)
            size_gt_hits_per_img_at_thr.append(gt_hits_list)

        size_bin_stats = size_binned_sensitivity(size_gt_areas_per_img_at_thr, size_gt_hits_per_img_at_thr, size_bins)

    # Bootstrap CIs (optional)
    ci_dict = _bootstrap_cis(per_image_by_thr, fppi_targets, len(img_ids),
                             iters=int(args.bootstrap), seed=int(args.bootstrap_seed))

    # YAML summary
    summary = {
        'num_images': len(img_ids),
        'min_area': int(args.min_area),
        'merge_radius': int(args.merge_radius),
        'froc_tp_by': args.froc_tp_by,
        'thresholds': [float(t) for t in thr_list],
        'results': results_by_thr,
        'fppi_targets': fppi_targets,
        'sensitivity_at_fppi': {str(k): float(v) for k, v in sens_at_fppi.items()},
        'froc_score_mean_sensitivity': float(froc_score),
        'operating_points_closest': operating_points,
        'precision_recall': {
            'thresholds': pr_pkg['thresholds'],
            'precision': pr_pkg['precision'],
            'recall': pr_pkg['recall'],
            'ap': pr_pkg['ap'],
            'best_f1': pr_pkg['best_f1'],
            'best_f1_thr': pr_pkg['best_f1_thr'],
            'precision_envelope': pr_pkg['precision_envelope'],
            'recall_sorted': pr_pkg['recall_sorted'],
        },
        'size_binned_sensitivity': {
            'bins': size_bin_stats.get('bins', []),
            'totals': size_bin_stats.get('totals', []),
            'hits': size_bin_stats.get('hits', []),
            'sensitivity': size_bin_stats.get('sensitivity', []),
            'evaluated_at_fppi': float(args.size_sens_at_fppi),
            'evaluated_at_thr': float(closest_thr) if closest_thr is not None else None,
        },
    }
    summary.update(ci_dict)

    with open(args.out, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False)
    print(f"\n💾 Saved metrics YAML to: {args.out}")

    # Optional per-image CSV
    if args.per_image_csv:
        with open(args.per_image_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['img_id', 'TP', 'FP', 'FN', 'N_gt'])
            # Choose a threshold (e.g., closest to size_sens_at_fppi) to dump; if not found, use first thr.
            dump_thr = closest_thr if closest_thr is not None else thr_list[0]
            rows = per_image_by_thr.get(dump_thr, [])
            for r in rows:
                w.writerow([r['img_id'], r['TP'], r['FP'], r['FN'], r['N_gt']])
        print(f"💾 Saved per-image breakdown (at thr={dump_thr}) to: {args.per_image_csv}")

    # Optional plots
    if args.plot:
        base = os.path.splitext(args.out)[0]

        # FROC
        xs = [r[0] for r in records]
        ys = [r[1] for r in records]
        plt.figure()
        plt.plot(xs, ys, marker='o')  # single plot, default colors
        if xs:
            plt.xlim(0, max(min(args.max_fppi, max(xs) * 1.05), 1.0))
        plt.ylim(0, 1.0)
        plt.xlabel("False Positives per Image (FPPI)")
        plt.ylabel("Sensitivity")
        plt.title("FROC")
        froc_png = base + "_froc.png"
        plt.savefig(froc_png, bbox_inches='tight', dpi=180)
        print(f"🖼  Saved FROC plot to: {froc_png}")

        # PR (use precision envelope vs sorted recall)
        plt.figure()
        if pr_pkg['recall_sorted']:
            plt.plot(pr_pkg['recall_sorted'], pr_pkg['precision_envelope'], marker='o')
        plt.xlim(0, 1.0)
        plt.ylim(0, 1.0)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision–Recall (AP={pr_pkg['ap']:.3f})")
        pr_png = base + "_pr.png"
        plt.savefig(pr_png, bbox_inches='tight', dpi=180)
        print(f"🖼  Saved PR plot to: {pr_png}")

if __name__ == "__main__":
    main()
