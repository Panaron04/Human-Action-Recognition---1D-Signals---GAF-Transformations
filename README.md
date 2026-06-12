# GAF-HAR — Time-Series Image Encodings for Human Activity Recognition

**Do Gramian Angular Fields actually help sensor-based HAR?** We benchmark GAF (and 6 other
time-series→image encodings) against strong 1D and classical baselines on **three public datasets**,
under a strict **subject-wise evaluation protocol** — and trace, step by step, what it takes to make
the 2D approach win.

> Course project — Computer Vision (CV26).
> Team: Karavas · Barkas · Evangelou · Lagou · Georgiou

---

## TL;DR — Findings

| # | Finding |
|---|---------|
| 1 | **Random train/test splits leak subject identity** and inflate accuracy by 10–30 points. All our numbers use subject-wise splits (the official protocols). |
| 2 | **GASF is actively harmful** (46% vs GADF's 75% on UTD-MHAD); combining them is worse than GADF alone. Our *Fusion-GADF* (same-axis + cross-axis GADF) keeps the fusion idea and drops GASF. |
| 3 | GAF images encode **subject-specific signatures** (sensor placement / personal style). Subject-wise validation + random 3D-rotation augmentation + test-time augmentation address this directly. |
| 4 | **ImageNet transfer learning is the bottleneck, not the GAF representation.** Replacing a pretrained ShuffleNetV2+Transformer with a compact **from-scratch CNN** (fewer parameters!) gains **+20 pts on UTD-MHAD and +14 pts on UCI HAR**. |
| 5 | With the right architecture, GAF **beats every baseline on UTD-MHAD (86.3%)** and **matches the 1D-CNN on UCI HAR (92.2% vs 92.4%)**. Encoding choice is scale-dependent: plain GADF wins on small data; cross-axis Fusion-GADF wins as data grows. |

Full experiment history with dated tables: see the `Progress_Log_*.pdf` reports.

## Results snapshot (test accuracy, subject-wise splits)

| Method | UTD-MHAD (27 cls) | UCI HAR (6 cls) |
|---|---|---|
| **GADF + Custom CNN (ours)** | **86.3 %** | 91.7 % |
| **Fusion-GADF + Custom CNN (ours)** | 83.7 % (85.1 % TTA) | **92.2 %** |
| 1D-CNN baseline | 83.0 % | 92.4 % |
| Random Forest (69 hand-crafted features) | 79.5 % | 90.4 % |
| Best GAF + pretrained backbone | 66.3 % | 78.5 % |

## What's inside

- **8+ encodings**, identical interface, fair single-variable comparison: GASF, GADF, GASF+GADF,
  MTF, Recurrence Plot, CWT scalogram, STFT spectrogram, **Fusion-24**, **Fusion-GADF**,
  **Fusion-GADF+magnitudes**.
- **4 architectures**: pretrained ShuffleNetV2+Transformer (`gaf2dnet`), from-scratch VGG-style CNN
  (`custom_cnn`), residual+SE+multi-dilation CNN (`custom_cnn_v2`), plus 1D-CNN / BiLSTM / Random
  Forest baselines.
- **3 datasets**: UTD-MHAD (inertial), UCI HAR, WISDM v1.1 — the last two **auto-download**.
- **Reproducible experiment framework**: every run writes `config.json`, `history.csv`,
  `metrics.json`, `best_model.pth`, training curves and confusion matrix to
  `results/<dataset>/<experiment>/`; completed experiments are auto-skipped (`SKIP_IF_DONE`);
  encodings are disk-cached; a final cell aggregates everything into
  `results/comparison/all_results.csv` + comparison charts.

## Quickstart

```bash
pip install torch torchvision numpy scipy scikit-learn pandas matplotlib seaborn tqdm PyWavelets requests
```

1. **UTD-MHAD** (manual download): get the Inertial `.mat` files from the
   [official page](https://personal.utdallas.edu/~kehtar/UTD-MHAD.html) (or a Kaggle mirror) and
   place them flat in `data/UTD_MHAD/Inertial/` (861 files, `a*_s*_t*_inertial.mat`).
   UCI HAR and WISDM download automatically on first run.
2. Open `HAR_GAF_Baselines_and_Encodings_v5.ipynb` and set the flags in `CONFIG` (first cell):
   `QUICK_MODE` (short vs full training), `RUN_UTD / RUN_UCI / RUN_WISDM`.
3. **Run All.** Finished experiments load from disk; only new ones train.

## Repository layout

```
├── HAR_GAF_Baselines_and_Encodings_v5.ipynb   # the full pipeline (latest)
├── data/                                       # datasets (auto-downloaded / manual UTD)
├── cache/                                      # encoded-image cache (safe to delete)
├── results/
│   ├── <DATASET>/<experiment>/                 # config, metrics, weights, figures per run
│   └── comparison/all_results.csv              # aggregated table + charts
├── figures/                                    # encoding galleries
└── Progress_Log_{1,2,3}.pdf                    # dated findings reports
```

## Evaluation protocol

- **UTD-MHAD**: train subjects {1,3,5}, validation subject {7} (entire, unseen), test {2,4,6,8}
  (the dataset's official odd/even protocol).
- **UCI HAR**: official subject-wise train/test split; 3 held-out training subjects for validation.
- **WISDM**: subjects 1–24 / 25–28 / 29–36.
- Model selection on validation only; test evaluated once with best-on-val weights.
- Training: AdamW, cosine schedule (optional warmup), label smoothing 0.1, random 3D-rotation +
  time-shift augmentation, optional test-time augmentation.

## References

- Wang & Oates, *Imaging Time-Series to Improve Classification and Imputation*, IJCAI 2015.
- Xu et al., *Human Activity Recognition Based on Gramian Angular Field and Deep CNN*, IEEE Access 2020.
- *GAFormer* (APSIPA 2023) — GADF-based HAR with attention backbones.
