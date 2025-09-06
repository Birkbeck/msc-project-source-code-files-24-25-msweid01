import argparse, os, random, yaml
import numpy as np
import torch
from torch.amp import autocast, GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import pandas as pd

from dataset import make_loaders
from archs import HybridEffB3ViTUNet
from loss import ComboLoss
from metrics import dice_coef, iou_score, roc_auc


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_ckpt(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def validate(model, loader, device, thr=0.5, limit=None):
    model.eval()
    dices, ious, aucs = [], [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            out = model(x)
            p = out.detach().cpu().numpy()
            g = y.detach().cpu().numpy()
            for pi, gi in zip(p, g):
                dices.append(dice_coef(pi[0], gi[0], thr=thr))
                ious.append(iou_score(pi[0], gi[0], thr=thr))
                aucs.append(roc_auc(pi[0], gi[0]))
            if limit is not None and (i + 1) >= limit:
                break
    return float(np.nanmean(dices)), float(np.nanmean(ious)), float(np.nanmean(aucs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt_dir", default="runs/fold1")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--balance", action="store_true")
    ap.add_argument("--pos_mult", type=float, default=8)

    ap.add_argument("--patch_prob", type=float, default=0.5)
    ap.add_argument("--patch_min", type=int, default=384)
    ap.add_argument("--patch_max", type=int, default=512)
    ap.add_argument("--patch_jitter", type=float, default=0.15)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = make_loaders(
        args.data_root, args.fold,
        batch_size=args.batch_size, num_workers=args.num_workers,
        balance=args.balance, pos_mult=args.pos_mult,
        imagenet_norm=False,
        crop_prob=args.patch_prob, crop_min=args.patch_min, crop_max=args.patch_max, pos_jitter=args.patch_jitter,
        sanity=True
    )

    model = HybridEffB3ViTUNet(num_classes=1).to(device)
    criterion = ComboLoss(
        pos_weight=450.0,  # tune (100–500) depending on imbalance
        alpha=0.2, beta=0.8,  # recall-friendly Tversky
        weights=(1.0, 2.0, 1.0),  # BCE, Dice, Tversky weights
        from_logits=False  # your model returns sigmoid probs
    )
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = GradScaler('cuda')
    best_dice = -1.0

    os.makedirs(args.ckpt_dir, exist_ok=True)
    log_csv = os.path.join(args.ckpt_dir, "train_log.csv")
    log_rows = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        running_loss = 0.0

        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                out = model(x)
                loss = criterion(out, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        # ---- Validation + logging
        val_dice, val_iou, val_auc = validate(model, val_loader, device, thr=args.thr)
        row = {
            "epoch": epoch,
            "train_loss": running_loss / len(train_loader),
            "val_dice": val_dice,
            "val_iou": val_iou,
            "val_auc": val_auc,
            "lr": scheduler.get_last_lr()[0],
        }
        log_rows.append(row)
        pd.DataFrame(log_rows).to_csv(log_csv, index=False)

        # Pretty summary line after each epoch
        tqdm.write(
            f"Epoch {epoch:>3}/{args.epochs} | "
            f"train_loss: {row['train_loss']:.4f} | "
            f"val_dice: {val_dice:.4f} | val_iou: {val_iou:.4f} | val_auc: {val_auc:.4f} | "
            f"lr: {row['lr']:.2e}"
        )

        # Checkpointing
        if val_dice > best_dice:
            best_dice = val_dice
            save_ckpt(model, os.path.join(args.ckpt_dir, "best_dice.pt"))
            tqdm.write(f"✅ New best Dice: {best_dice:.4f} — saved to best_dice.pt")
        save_ckpt(model, os.path.join(args.ckpt_dir, "last.pt"))

    with open(os.path.join(args.ckpt_dir, "summary.yaml"), "w") as f:
        yaml.safe_dump({"best_val_dice": float(best_dice), "epochs": args.epochs}, f)


if __name__ == "__main__":
    main()
