import argparse, glob, json, math, os, re
from typing import Dict, Any
import numpy as np
import pandas as pd
import yaml

FPPI_KEYS = ["0.125", "0.25", "0.5", "1.0", "2.0", "4.0", "8.0"]

def _read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _extract_fold_id(eval_dir: str) -> str:
    m = re.search(r"fold(\d+).*_eval", os.path.basename(eval_dir))
    if m: return m.group(1)
    for part in eval_dir.split(os.sep):
        m = re.search(r"fold(\d+)", part)
        if m: return m.group(1)
    return os.path.basename(eval_dir)

def _fmt(mean: float, std: float, ndigits: int = 3) -> str:
    if math.isnan(mean) or math.isnan(std):
        return "NA"
    return f"{mean:.{ndigits}f} ± {std:.{ndigits}f}"

def aggregate(eval_glob: str, out_dir: str, fixed_thr: float = 0.5) -> None:
    os.makedirs(out_dir, exist_ok=True)
    eval_dirs = sorted(glob.glob(eval_glob))
    if not eval_dirs:
        raise SystemExit(f"No eval dirs match glob: {eval_glob}")

    rows = []
    for ed in eval_dirs:
        metrics_p = os.path.join(ed, "metrics.yaml")
        summary_p = os.path.join(ed, "summary.yaml")
        if not os.path.exists(metrics_p):
            print(f"[WARN] metrics.yaml missing in {ed}, skipping.")
            continue
        m = _read_yaml(metrics_p)
        fold_id = _extract_fold_id(ed)

        cpm = float(m.get("froc_score_mean_sensitivity", float("nan")))
        ap = float((m.get("precision_recall") or {}).get("ap", float("nan")))
        num_images = m.get("num_images")
        froc_tp_by = m.get("froc_tp_by", "pred")

        sens_map = {str(k): float(v) for k, v in (m.get("sensitivity_at_fppi") or {}).items()}
        for k in FPPI_KEYS:
            sens_map.setdefault(k, float("nan"))

        dice_pos = iou_pos = auc_pos = float("nan")
        if os.path.exists(summary_p):
            s = _read_yaml(summary_p)
            ref = s.get("metrics_at_ref_thr", {})
            pos = ref.get("positives_only", {})
            dice_pos = float(pos.get("dice_mean", float("nan")))
            iou_pos  = float(pos.get("iou_mean", float("nan")))
            auc_pos  = float(pos.get("auc_mean", float("nan")))

        rows.append(dict(
            fold=str(fold_id),
            cpm=cpm,
            ap=ap,
            sens_0p125=sens_map["0.125"],
            sens_0p25=sens_map["0.25"],
            sens_0p5=sens_map["0.5"],
            sens_1p0=sens_map["1.0"],
            sens_2p0=sens_map["2.0"],
            sens_4p0=sens_map["4.0"],
            sens_8p0=sens_map["8.0"],
            dice_pos_0p5=dice_pos,
            iou_pos_0p5=iou_pos,
            auc_pos_0p5=auc_pos,
            num_images=num_images,
            froc_tp_by=froc_tp_by,
            eval_dir=ed,
        ))

    if not rows:
        raise SystemExit("No valid metrics found.")

    df = pd.DataFrame(rows).sort_values("fold")
    df_path = os.path.join(out_dir, "fold_metrics.csv")
    df.to_csv(df_path, index=False)

    agg_fields = [
        "cpm","ap",
        "sens_0p125","sens_0p25","sens_0p5","sens_1p0","sens_2p0","sens_4p0","sens_8p0",
        "dice_pos_0p5","iou_pos_0p5","auc_pos_0p5"
    ]
    mean = df[agg_fields].mean(numeric_only=True)
    std  = df[agg_fields].std(ddof=1, numeric_only=True)

    summary = {
        "n_folds": int(len(df)),
        "eval_glob": eval_glob,
        "fixed_threshold_for_segmentation": fixed_thr,
        "froc_tp_by": list(df["froc_tp_by"].unique()),
        "means": {k: float(v) for k, v in mean.items()},
        "stds":  {k: float(v) for k, v in std.items()},
        "per_fold": df.to_dict(orient="records"),
    }
    with open(os.path.join(out_dir, "cv_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # CSV and Markdown summary table
    def row(label, key): return {"Metric": label, "Mean ± SD": _fmt(mean[key], std[key])}
    table = [
        row("CPM (mean sens over FPPI 0.125–8)", "cpm"),
        row("Average Precision (lesion PR)", "ap"),
        row("Sensitivity @ 0.125 FPPI", "sens_0p125"),
        row("Sensitivity @ 0.25 FPPI", "sens_0p25"),
        row("Sensitivity @ 0.5 FPPI", "sens_0p5"),
        row("Sensitivity @ 1.0 FPPI", "sens_1p0"),
        row("Sensitivity @ 2.0 FPPI", "sens_2p0"),
        row("Sensitivity @ 4.0 FPPI", "sens_4p0"),
        row("Sensitivity @ 8.0 FPPI", "sens_8p0"),
        row("Positives-only Dice @ thr=0.5", "dice_pos_0p5"),
        row("Positives-only IoU @ thr=0.5", "iou_pos_0p5"),
        row("Pixel AUC (positives)", "auc_pos_0p5"),
    ]
    tab_df = pd.DataFrame(table)
    tab_df.to_csv(os.path.join(out_dir, "cv_table.csv"), index=False)

    md = ["| Metric | Mean ± SD |", "|:--|:--|"] + [f"| {r['Metric']} | {r['Mean ± SD']} |" for _, r in tab_df.iterrows()]
    with open(os.path.join(out_dir, "cv_table.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_glob", required=True, help='Glob for eval dirs, e.g. "runs/fold*_e60_eval"')
    ap.add_argument("--out", required=True, help="Output dir for summaries")
    ap.add_argument("--fixed_thr", type=float, default=0.5, help="Fixed thr for seg metrics (reads metrics_at_ref_thr)")
    args = ap.parse_args()
    aggregate(args.eval_glob, args.out, args.fixed_thr)
