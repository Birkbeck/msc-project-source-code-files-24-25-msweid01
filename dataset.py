# dataset.py
import os, glob, random, math
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def _list_images_masks(img_dir, msk_dir):
    import glob, os

    imgs_all = [p for p in glob.glob(os.path.join(img_dir, "*")) if os.path.isfile(p)]
    msks_all = [p for p in glob.glob(os.path.join(msk_dir, "*")) if os.path.isfile(p)]

    IMG_EXTS = {".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp",".pnm",".pgm"}
    MSK_EXTS = {".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp",".pnm",".pgm"}

    def _stem(p):
        return os.path.splitext(os.path.basename(p))[0]

    def _norm_stem(s):
        s = s.lower()
        for suf in ["_mask","-mask","_gt","-gt","_label","-label","_labels","-labels",
                    "_anno","-anno","_ann","-ann"]:
            if s.endswith(suf):
                s = s[: -len(suf)]
        return s

    # index masks by normalized stem
    masks_by_norm = {}
    masks_by_exact = {}
    for mp in msks_all:
        ext = os.path.splitext(mp)[1].lower()
        if ext and ext not in MSK_EXTS:
            pass
        exact = _stem(mp)
        norm  = _norm_stem(exact)
        masks_by_exact.setdefault(exact, []).append(mp)
        masks_by_norm.setdefault(norm,  []).append(mp)

    img_paths, msk_paths = [], []

    for ip in sorted(imgs_all):
        ext = os.path.splitext(ip)[1].lower()
        if ext and ext not in IMG_EXTS:
            pass
        s_exact = _stem(ip)
        s_norm  = _norm_stem(s_exact)

        cand = None
        # 1) try normalized stem
        if s_norm in masks_by_norm and len(masks_by_norm[s_norm]):
            cand = masks_by_norm[s_norm][0]
        # 2) exact stem
        elif s_exact in masks_by_exact and len(masks_by_exact[s_exact]):
            cand = masks_by_exact[s_exact][0]
        else:
            # 3) fallback
            pref = [m for m in msks_all if os.path.basename(m).startswith(s_exact)]
            if len(pref):
                cand = pref[0]

        if cand is not None and os.path.exists(cand):
            img_paths.append(ip)
            msk_paths.append(cand)

    return img_paths, msk_paths


def _to_rgb_from_green(img_bgr):
    # extract green channel, duplicate to 3-ch
    g = img_bgr[:, :, 1]
    g = np.clip(g, 0, 255).astype(np.uint8)
    g3 = np.stack([g, g, g], axis=-1)
    return g3

def _read_mask_gray(path):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    m = (m > 0).astype(np.uint8)  # binarize
    return m

def _safe_crop(x, y, cx, cy, size):
    H, W = x.shape[:2]
    half = size // 2
    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1 = min(W, x0 + size)
    y1 = min(H, y0 + size)
    # adjust
    x0 = max(0, x1 - size)
    y0 = max(0, y1 - size)
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    return x[y0:y1, x0:x1], y[y0:y1, x0:x1]

class EOPHTADataset(Dataset):
    def __init__(self,
                 img_dir, msk_dir,
                 size=768,
                 augment=True,
                 imagenet_norm=False,
                 crop_prob=0.5,
                 crop_min=384,
                 crop_max=512,
                 pos_jitter=0.15,
                 sanity=False):
        self.img_paths, self.msk_paths = _list_images_masks(img_dir, msk_dir)
        assert len(self.img_paths) == len(self.msk_paths) and len(self.img_paths) > 0, \
            (f"No pairs in {img_dir} vs {msk_dir}. "
             f"Found {len(self.img_paths)} images, {len(self.msk_paths)} masks. "
             f"Please check extensions/suffixes (e.g., *_mask, *_gt).")

        f"No pairs in {img_dir} vs {msk_dir}"
        self.size = int(size)
        self.augment = augment
        self.imagenet_norm = imagenet_norm
        self.crop_prob = float(crop_prob)
        self.crop_min = int(crop_min)
        self.crop_max = int(crop_max)
        self.pos_jitter = float(pos_jitter)
        self.sanity = sanity
        self._build_transforms()

    def _build_transforms(self):
        mean, std = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) if self.imagenet_norm else ((0.0, 0.0, 0.0),
                                                                                               (1.0, 1.0, 1.0))
        if self.augment:
            self.tf = A.Compose([
                A.Resize(self.size, self.size, interpolation=cv2.INTER_CUBIC),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ])
        else:
            self.tf = A.Compose([
                A.Resize(self.size, self.size, interpolation=cv2.INTER_CUBIC),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i):
        img_path = self.img_paths[i]
        msk_path = self.msk_paths[i]

        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(img_path)
        img = _to_rgb_from_green(img_bgr)
        msk = _read_mask_gray(msk_path)

        H, W = msk.shape
        use_crop = False
        if self.crop_prob > 0 and (msk > 0).any() and random.random() < self.crop_prob:
            use_crop = True
            # choose crop size
            cs = random.randint(self.crop_min, self.crop_max)
            ys, xs = np.where(msk > 0)
            k = random.randrange(len(xs))
            cx, cy = int(xs[k]), int(ys[k])
            # jitter
            j = int(self.pos_jitter * cs)
            cx = np.clip(cx + random.randint(-j, j), 0, W - 1)
            cy = np.clip(cy + random.randint(-j, j), 0, H - 1)
            img, msk = _safe_crop(img, msk, cx, cy, cs)

        # scale to [0,1] float before normalize
        img = img.astype(np.float32) / 255.0
        msk = msk.astype(np.uint8)

        out = self.tf(image=img, mask=msk)
        x = out["image"]
        y = out["mask"].unsqueeze(0).float()

        if self.sanity and random.random() < 0.003:
            mn, mx = float(x.min()), float(x.max())
            print(f"[sanity] tensor range after Normalize: min={mn:.4f}, max={mx:.4f} (crop={use_crop})")
        return x, y


def _resolve_split_dirs(data_root, fold, split):
    base = os.path.join(data_root, f"fold{fold}")

    cand1_img = os.path.join(base, split, "images")
    cand1_msk = os.path.join(base, split, "masks")
    if os.path.isdir(cand1_img) and os.path.isdir(cand1_msk):
        return cand1_img, cand1_msk

    cand2_img = os.path.join(base, f"{split}_images")
    cand2_msk = os.path.join(base, f"{split}_masks")
    if os.path.isdir(cand2_img) and os.path.isdir(cand2_msk):
        return cand2_img, cand2_msk

    tried = [
        os.path.abspath(cand1_img),
        os.path.abspath(cand1_msk),
        os.path.abspath(cand2_img),
        os.path.abspath(cand2_msk),
    ]
    raise FileNotFoundError(
        f"Could not find {split} image/mask folders for fold {fold}.\n"
        f"Tried:\n  - {tried[0]}\n  - {tried[1]}\n  - {tried[2]}\n  - {tried[3]}"
    )

def make_loaders(data_root, fold,
                 batch_size=2, num_workers=0,
                 balance=False, pos_mult=8,
                 imagenet_norm=False,
                 crop_prob=0.5, crop_min=384, crop_max=512, pos_jitter=0.15,
                 sanity=False):
    tr_img, tr_msk = _resolve_split_dirs(data_root, fold, "train")
    va_img, va_msk = _resolve_split_dirs(data_root, fold, "val")

    train_ds = EOPHTADataset(tr_img, tr_msk,
                             size=768, augment=True,
                             imagenet_norm=imagenet_norm,
                             crop_prob=crop_prob, crop_min=crop_min, crop_max=crop_max,
                             pos_jitter=pos_jitter, sanity=sanity)
    val_ds   = EOPHTADataset(va_img, va_msk,
                             size=768, augment=False,
                             imagenet_norm=imagenet_norm,
                             crop_prob=0.0, crop_min=crop_min, crop_max=crop_max,
                             pos_jitter=pos_jitter, sanity=sanity)

    if balance:
        weights = []
        for _, y in train_ds:
            pos = float(y.sum().item() > 0.5)
            weights.append(pos_mult if pos > 0 else 1.0)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                  num_workers=num_workers, pin_memory=True, drop_last=False)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True, drop_last=False)

    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=num_workers, pin_memory=True, drop_last=False)

    return train_loader, val_loader


def make_loaders_cv3way(data_root, fold,
                        inner_val_ratio=0.125, inner_val_seed=123,
                        batch_size=2, num_workers=0,
                        balance=False, pos_mult=8,
                        imagenet_norm=False,
                        crop_prob=0.5, crop_min=384, crop_max=512, pos_jitter=0.15,
                        sanity=False):

    # fold train (pool) and fold val (test)
    tr_img_dir, tr_msk_dir = _resolve_split_dirs(data_root, fold, "train")
    te_img_dir, te_msk_dir = _resolve_split_dirs(data_root, fold, "val")

    # the pool to be splitted
    pool_ds = EOPHTADataset(tr_img_dir, tr_msk_dir,
                            size=768, augment=True,
                            imagenet_norm=imagenet_norm,
                            crop_prob=crop_prob, crop_min=crop_min, crop_max=crop_max,
                            pos_jitter=pos_jitter, sanity=sanity)

    # build pos/neg flags
    pos_flags = []
    for mp in pool_ds.msk_paths:
        g = _read_mask_gray(mp)
        pos_flags.append(int((g > 0).any()))
    idx_pos = [i for i, f in enumerate(pos_flags) if f == 1]
    idx_neg = [i for i, f in enumerate(pos_flags) if f == 0]

    rng = random.Random(inner_val_seed)
    n_pos = len(idx_pos)
    n_neg = len(idx_neg)
    n_val_pos = max(1, int(round(inner_val_ratio * n_pos)))
    n_val_neg = max(1, int(round(inner_val_ratio * n_neg)))

    idx_pos_shuf = idx_pos[:]; rng.shuffle(idx_pos_shuf)
    idx_neg_shuf = idx_neg[:]; rng.shuffle(idx_neg_shuf)

    inner_val_idx = idx_pos_shuf[:n_val_pos] + idx_neg_shuf[:n_val_neg]
    inner_train_idx = idx_pos_shuf[n_val_pos:] + idx_neg_shuf[n_val_neg:]
    rng.shuffle(inner_val_idx)
    rng.shuffle(inner_train_idx)

    inner_train_ds = Subset(pool_ds, inner_train_idx)
    inner_val_ds   = Subset(pool_ds, inner_val_idx)

    # test loader uses augment=False and no crops
    test_ds = EOPHTADataset(te_img_dir, te_msk_dir,
                            size=768, augment=False,
                            imagenet_norm=imagenet_norm,
                            crop_prob=0.0, crop_min=crop_min, crop_max=crop_max,
                            pos_jitter=pos_jitter, sanity=sanity)

    # loaders
    if balance:
        w = [pos_mult if pos_flags[i] == 1 else 1.0 for i in inner_train_idx]
        sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
        inner_train_loader = DataLoader(inner_train_ds, batch_size=batch_size, sampler=sampler,
                                        num_workers=num_workers, pin_memory=True, drop_last=False)
    else:
        inner_train_loader = DataLoader(inner_train_ds, batch_size=batch_size, shuffle=True,
                                        num_workers=num_workers, pin_memory=True, drop_last=False)

    inner_val_loader = DataLoader(inner_val_ds, batch_size=1, shuffle=False,
                                  num_workers=num_workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=num_workers, pin_memory=True, drop_last=False)

    return inner_train_loader, inner_val_loader, test_loader
