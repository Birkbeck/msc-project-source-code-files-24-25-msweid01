import argparse
import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".pgm", ".ppm"}

# Suffixes sometimes present on mask filenames (kept here for robustness if needed later)
SUFFIXES_TO_STRIP = [
    "_mask", "-mask", "_ma", "-ma", "_lesion", "-lesion", "_gt", "-gt",
    "_anno", "-anno", "_annotation", "-annotation"
]

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def load_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img

def load_mask(path):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    return m

def preprocess_image(img, size):
    green = img[:, :, 1]
    green = cv2.resize(green, (size, size), interpolation=cv2.INTER_AREA)
    green = green.astype(np.float32) / 255.0
    img3 = np.stack([green, green, green], axis=2)
    return (img3 * 255.0).astype(np.uint8)

def preprocess_mask(mask, size):
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0).astype(np.uint8) * 255
    return mask

def normalize_stem(p: Path) -> str:
    b = p.stem
    bl = b.lower()
    for suf in SUFFIXES_TO_STRIP:
        if bl.endswith(suf):
            b = b[: -len(suf)]
            bl = b.lower()
    return bl

def rglob_images(root: Path):
    return [p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in IMG_EXTS and p.name.lower() != "thumbs.db"]

def build_index(files):
    idx = {}
    for p in files:
        key = normalize_stem(p)
        idx.setdefault(key, []).append(p)
    return idx

def find_pairs(img_dir, mask_dir, healthy_dir=None):
    img_root = Path(img_dir)
    msk_root = Path(mask_dir)

    imgs = rglob_images(img_root)
    msks = rglob_images(msk_root)

    img_idx = build_index(imgs)
    msk_idx = build_index(msks)

    pairs = []
    pos_count = 0
    for key, img_list in img_idx.items():
        if key in msk_idx:
            mpath = msk_idx[key][0]       # use first matching mask
            for ipath in img_list:
                pairs.append((ipath, mpath))  # positive sample
                pos_count += 1

    neg_count = 0
    if healthy_dir is not None:
        healthy_root = Path(healthy_dir)
        healthy_imgs = rglob_images(healthy_root)
        for ipath in healthy_imgs:
            pairs.append((ipath, None))   # negative sample (no mask; will create blank)
            neg_count += 1

    if not pairs:
        # Diagnostics to help
        sample_imgs = [p.name for p in imgs[:15]]
        sample_msks = [p.name for p in msks[:15]]
        print("Diagnostics — example files:")
        print("Images:", sample_imgs)
        print("Masks :", sample_msks)

    return pairs, pos_count, neg_count

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True, help="Root to MA images (e.g., ...\\e_optha_MA\\MA)")
    ap.add_argument("--mask_dir", required=True, help="Root to Annotation_MA masks (e.g., ...\\e_optha_MA\\Annotation_MA)")
    ap.add_argument("--healthy_dir", default=None, help="Optional root to healthy images (no masks).")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--k_folds", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pairs, pos_count, neg_count = find_pairs(args.img_dir, args.mask_dir, args.healthy_dir)
    if not pairs:
        raise RuntimeError("No image-mask pairs found after recursive matching. Check paths (use --img_dir MA and --mask_dir Annotation_MA).")

    print(f"Found {pos_count} positive (with masks) and {neg_count} negative (healthy) images. Total: {len(pairs)}")

    # Labels for stratification: 1=lesion image, 0=healthy
    labels = []
    for _, m in tqdm(pairs, desc="Scanning masks / labeling"):
        if m is None:
            labels.append(0)
        else:
            mask = load_mask(m)
            labels.append(int(mask.max() > 0))

    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    idxs = np.arange(len(pairs))

    for fold, (tr, va) in enumerate(skf.split(idxs, labels), start=1):
        fold_dir = os.path.join(args.out_dir, f"fold{fold}")
        ti = os.path.join(fold_dir, "train_images")
        tm = os.path.join(fold_dir, "train_masks")
        vi = os.path.join(fold_dir, "val_images")
        vm = os.path.join(fold_dir, "val_masks")
        for d in [ti, tm, vi, vm]:
            ensure_dir(d)

        print(f"Processing fold {fold} with {len(tr)} train and {len(va)} val images")
        for split, id_list, odir_img, odir_msk in [("train", tr, ti, tm), ("val", va, vi, vm)]:
            for i in tqdm(id_list, desc=f"Fold{fold} {split}"):
                ipath, mpath = pairs[i]
                img = load_image(ipath)
                img_p = preprocess_image(img, args.size)

                if mpath is None:
                    # healthy -> blank mask
                    msk_p = np.zeros((args.size, args.size), dtype=np.uint8)
                else:
                    msk = load_mask(mpath)
                    msk_p = preprocess_mask(msk, args.size)

                stem = Path(ipath).stem
                cv2.imwrite(os.path.join(odir_img, stem + ".png"), img_p)
                cv2.imwrite(os.path.join(odir_msk, stem + ".png"), msk_p)

    print("Done.")

if __name__ == "__main__":
    main()
