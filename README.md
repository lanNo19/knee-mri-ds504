# knee-mri-ds504

DS 504/704 Data Mining, team T2 (Ablay Abdimalinov, Dinmukhammed Sagatbekov, Temirlan Yessembayev).

Question: when the only training labels come from multilingual radiology reports, which frozen
image encoder transfers better to knee-MRI sites never seen in training: a self-supervised ViT
(DINOv2 ViT-S/14) or a supervised CNN (ConvNeXt-Tiny)? Data: RSNA Knee Abnormality Detection
(Kaggle, 2026); second dataset: KneeMRI (Štajduhar et al. 2017, CC BY-NC-ND 4.0).

## What is in this repo

Code and configs only. No data, UIDs, labels, folds, predictions, weights or report text:
those stay in private Kaggle outputs (competition data-use terms).

```
src/knee/            package: safe logging, DICOM reading, series choice, labels, groups, folds, caches
src/knee/nb/         one entry point per notebook (smoke, n0, n3, n4, ...)
notebooks/RUNBOOK.md the two cells of every notebook, its settings, inputs and pass bands
tests/               tests on synthetic data only
```

## Tests

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]" pillow
python -m pytest -q
```

## Notebooks

Each Kaggle notebook installs this package at a pinned commit and runs one step; see
notebooks/RUNBOOK.md. Errors print only their type and location (knee.safe_log), and each
step's output is hidden except for its summary, so no data values appear in notebook output.
