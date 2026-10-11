# Predicting a Decline in Patient-Reported Outcomes for Cancer Patients on Chemotherapy

UE24CS352A Machine Learning - Mini-Project (Team 8)

Team members:
- Rushikesh - PES1UG24AM394
- Prajwal - PES1UG24AM386

Re-implementation of the problem in Ostberg & Peterson, *Predicting a Decline in Patient Reported
Outcomes for Cancer Patients on Chemotherapy* (Stanford CS229, Spring 2020).

> **Data note.** The original data (Stanford STARR EHR warehouse) is private. This repo uses a
> **synthetic cohort** that mirrors the paper's structure (1,252 patients, same feature groups,
> same two outcome definitions). Results show the **pipeline works**, not clinical findings.

## Task
Using only information available **before the first chemotherapy dose** (demographics, vitals, labs,
diagnoses, medications, procedures, utilisation, baseline PROMIS scores), predict whether a patient's
PROMIS Global Physical Health (GPH) score will decline on chemotherapy:

| Outcome | Definition |
|---|---|
| `y_drop` | GPH on chemo - baseline GPH <= -10 (a 1-SD drop) |
| `y_threshold` | GPH on chemo <= 40 (1 SD below the population mean of 50) |

## Setup
```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python src/generate_data.py      # creates data/synthetic_cohort.csv
python src/train.py              # full experiments (about 10-15 min on 1 CPU)
python src/train.py --quick      # fast smoke test (single lab scheme, about 3 min)
python src/demo.py               # live demo: risk scores for held-out patients
python src/demo.py --index 3     # detailed view of one patient
```

## Repository layout
```
src/generate_data.py   synthetic cohort generator (informative lab missingness, regression-to-mean)
src/common.py          feature engineering (LabFeaturizer), model zoo + grids, ensemble, bootstrap CI
src/train.py           Experiment 1 (models x schemes x outcomes), Experiment 2 (ablation), plots
src/demo.py            loads the saved ensemble and scores patients
results/               CSV tables, ROC curves, ablation plot, training log
models/                saved best ensemble (joblib)
```

## Approach
- **Lab featurisation (3 schemes):** `binned` (tertiles + explicit missing bin), `knn` (KNN-imputed
  continuous values), `hybrid` (KNN-imputed values **plus** missingness flags - the paper's suggested
  future work).
- **Models:** LASSO / Ridge / Elastic-Net logistic regression, KNN, Random Forest, Gradient Boosting
  (sklearn HistGradientBoosting), SVM, MLP, and a soft-voting ensemble of the six strongest.
- **Tuning:** grid search, 5-fold stratified CV on the training set, scored by AUROC. 80/20 train/test split.
- **Experiment 2:** drop one feature group at a time from the best ensemble (drop outcome); 95% bootstrap CIs.

## Deliberate differences from the original report
1. **No test-set leakage in tuning.** The report used the test set for early stopping (LightGBM, MLP).
   Here early stopping uses an internal validation split, and imputation/binning are fit on training folds only.
2. **Label sign.** The report's formula says `change <= 10`; the intended meaning (a drop of >= 10) is `change <= -10`.
3. **Smaller grids / 5-fold CV** to fit a one-week project on limited hardware.
4. Added the `hybrid` lab scheme and a baseline-scores-only reference.

## Results (synthetic data, held-out test set of 251 patients)
| Outcome | Best ensemble scheme | Test AUROC (95% CI) |
|---|---|---|
| `y_drop` (10-point drop) | binned | 0.737 (0.673 - 0.800) |
| `y_threshold` (GPH <= 40) | knn / hybrid | 0.902 (about 0.86 - 0.93) |

Ablation (drop outcome): removing labs (0.695) or baseline scores (0.672) hurts most; other groups
change AUROC by less than 0.015 with overlapping confidence intervals.
These numbers describe the synthetic generator, not real patients.

Full tables and plots: `results/experiment1_all_models.csv`, `results/experiment2_ablation.csv`,
`results/roc_curves.png`, `results/ablation.png`. Re-running `train.py` regenerates them.
