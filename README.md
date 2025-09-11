# Microaneurysm Segmentation on E-Ophtha MA

A clean, reproducible pipeline for lesion-level microaneurysm (MA) segmentation on the **E-Ophtha MA** dataset using a **Hybrid EfficientNet-B3 encoder + MobileViT-U-Net decoder**. The repo includes dataset preparation, k-fold training, pixel and lesion metrics, FROC analysis, and overlay visualizations.

---

## Dataset

* Source: **E-Ophtha MA** from ADCIS and OPHDIAT. Request access and download here: [https://www.adcis.net/en/third-party/e-ophtha/](https://www.adcis.net/en/third-party/e-ophtha/)
* After downloading, keep the original image and mask structure. Healthy images are used for negative patches.

---

## Environment

```bash
conda create -n eo-ma python=3.11 -y
conda activate eo-ma
pip install -r requirements.txt
```

---

## Prepare folds

Create processed inputs and k-fold splits.

**Example**

```bash
python prepare_eophta.py \
  --img_dir "path\to\e_ophtha_MA\MA" \
  --mask_dir "path\to\e_ophtha_MA\Annotation_MA" \
  --healthy_dir "path\to\e_ophtha_MA\healthy" \
  --out_dir "inputs\e_ophtha_MA" \
  --size 768 \
  --k_folds 4 \
  --seed 42
```

Notes:

* `--size` controls the processed tile size for training and validation.
* Output layout will include `fold{0..k-1}` subfolders with `train_images`, `train_masks`, `val_images`, and `val_masks`.

---

## Train

Train on a chosen fold with patch sampling and class balancing.

**Example**

```bash
python train.py \
  --data_root "inputs\e_ophtha_MA" \
  --fold 1 \
  --epochs 20 \
  --batch_size 2 \
  --lr 1e-4 \
  --ckpt_dir "runs\fold1_combo_patch" \
  --num_workers 0 \
  --balance \
  --pos_mult 12 \
  --patch_prob 0.6 \
  --patch_min 384 \
  --patch_max 544 \
  --patch_jitter 0.2
```

Key switches:

* `--balance` uses a ComboLoss configuration with positive reweighting. `--pos_mult` scales the positive class.
* Patch controls define a random crop pipeline around lesions and background.

---

## Validate and threshold sweep

Run pixel and lesion evaluations on the validation split for the same fold. This also saves probability maps and PNGs for later FROC analysis.

**Example**

```bash
python val.py \
  --data_root "inputs\e_ophtha_MA" \
  --fold 1 \
  --ckpt "runs\fold1_combo_patch\best_dice.pt" \
  --out "runs\fold1_combo_patch_eval_hiThr" \
  --num_workers 0 \
  --save_probs --save_png --save_preds \
  --merge_radius 4 --min_area 4 --open_k 0 --close_k 1 \
  --min_circ 0.2 --min_sol 0.8 --filter_border \
  --scan_min 0.50 --scan_max 0.95 --scan_steps 20
```

Post-processing flags:

* `--merge_radius` merges nearby blobs before lesion counting.
* `--min_area`, `--min_circ`, `--min_sol` filter spurious detections.
* `--scan_*` defines a threshold sweep for PR and ROC curves.

---

## Report and FROC

Compute pixel metrics, lesion metrics, and FROC/CPM at standard FP per image points. Point the script to the validation set and the directory of saved probability maps from the previous step.

**Example**

```bash
python report_metrics.py \
  --img_dir "inputs\e_ophtha_MA\fold3\val_images" \
  --gt_dir  "inputs\e_ophtha_MA\fold3\val_masks" \
  --prob_dir "runs\fold1on3_eval\probs_png" \
  --thr_sweep "0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8,0.9,0.95,0.975,0.99" \
  --min_area 4 \
  --merge_radius 4 \
  --fppi_points "0.125,0.25,0.5,1,2,4,8" \
  --plot \
  --out "runs\fold1on3_eval\metrics.yaml"
```

Outputs include a YAML summary, CSV tables, and figures for Dice, IoU, PR, ROC, and FROC. CPM is computed at the FP per image points in `--fppi_points`.

---

## Model and loss

* **Architecture**: EfficientNet-B3 encoder with MobileViT-U-Net style decoder with skip connections.
* **Loss**: ComboLoss that mixes BCE, Dice, and Tversky. Tversky parameters are configurable in the training script.

---

## Results template

Replace the values with your own after running cross-validation.

| Metric                 | Mean ± Std (k-fold) |
| ---------------------- | ------------------- |
| Dice @ 0.5             | 0.xx ± 0.xx         |
| IoU @ 0.5              | 0.xx ± 0.xx         |
| ROC-AUC                | 0.xx ± 0.xx         |
| CPM                    | 0.xx                |
| Sensitivity @ 1 FP/img | 0.xx                |

All figures and tables are produced under `runs/` or `reports/` depending on your output folders.

---

## Reproducibility tips

* Set `--seed` in all scripts to fix the fold splits and training order.
* Keep `--num_workers 0` on Windows if you face DataLoader issues. Increase on Linux if possible.
* Always report fixed operating points together with threshold sweeps for fair comparison.


