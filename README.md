# Predicting a Decline in Patient-Reported Outcomes for Cancer Patients on Chemotherapy

**Course:** UE24CS352A Machine Learning — Mini-Project (Team 8)

## Team Members

| Name | SRN |
|---|---|
| Rushikesh | PES1UG24AM394 |
| Prajwal | PES1UG24AM386 |

This project explores whether patient-reported outcomes (PROs) can be used to predict a decline in health-related quality of life during chemotherapy. It is inspired by Ostberg and Peterson, *Predicting a Decline in Patient Reported Outcomes for Cancer Patients on Chemotherapy* (Stanford CS229, Spring 2020).

> **Data and interpretation note:** The original Stanford STARR EHR data is private. The repository's synthetic cohort and any separately supplied cleaned dataset are not the original clinical cohort. Results from synthetic or small-sample data demonstrate an experimental pipeline; they must not be presented as validated clinical findings or used for medical decisions.

## Prediction task

The project studies two binary outcomes based on baseline and on-treatment overall PRO scores:

| Outcome | Definition used by the notebook |
|---|---|
| `drop` | `Overall_end - Overall_baseline <= -10` — a decline of at least 10 points |
| `threshold` | `Overall_end <= mean(Overall_baseline) - std(Overall_baseline)` — an end score below the dataset's baseline mean-minus-one-standard-deviation threshold |

The threshold above is calculated from the loaded dataset. It may differ from the fixed score threshold used in the original paper, so describe it accurately when reporting results.

## Project structure

Place the notebook and dataset as shown below. The `outfolder/` directory is created automatically when the notebook runs.

```
main/
├── README.md
├── data/
│   ├── breast_pro_clean.csv          # cleaned dataset used by the notebook
│   └── synthetic_cohort.csv          # optional dataset for the original src pipeline
├── model/
│   └── ml_experiments.ipynb          # notebook-based experiments
├── src/
│   ├── generate_data.py              # synthetic cohort generator, if included
│   ├── common.py                     # feature engineering and model helpers, if included
│   ├── train.py                      # script-based experiments, if included
│   └── demo.py                       # demo, if included
└── outfolder/                        # notebook output files; generated automatically
```

Keep `breast_pro_clean.csv` in `main/data/` and `ml_experiments.ipynb` in `main/model/`. The notebook searches the current directory and its parent directories for the `data/` folder, so it can be launched from either `main/` or `main/model/`. Its generated files are written to `main/outfolder/`.

## Environment setup

Use Python 3.10 or newer where possible. From the project root, create and activate a virtual environment.

**Windows PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install pandas numpy matplotlib scikit-learn joblib lightgbm jupyter
```

**macOS / Linux**

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pandas numpy matplotlib scikit-learn joblib lightgbm jupyter
```

If the repository has a maintained `requirements.txt`, you can install its dependencies with `python -m pip install -r requirements.txt` as well.

## Run the notebook experiments

1. Open a terminal in `main/` or `main/model/`.
2. Start Jupyter:
   ```bash
   python -m jupyter notebook
   ```
3. Open `model/ml_experiments.ipynb` (or open the notebook directly if Jupyter starts in `model/`).
4. Run the cells from top to bottom.
5. Check `outfolder/` for saved results.

The notebook checks the dataset, engineers baseline PRO, clinical, and socio-demographic features, compares models, and optionally performs feature-group ablation and LASSO coefficient analysis.

## Models and evaluation

The notebook includes LASSO, Ridge, Elastic Net, K-nearest neighbors, LightGBM, support vector machine, Random Forest, feed-forward neural network (FFNN), and a soft-voting ensemble. It compares KNN imputation with median imputation plus missingness indicators. Hyperparameters are tuned inside the training folds of nested stratified cross-validation; held-out outer folds are not used for tuning.

In the notebook's configuration cell:

- `REPEATS = 1` runs one repetition of 5-fold outer cross-validation for an initial run. Set it to `3` for a more stable repeated evaluation, at additional computational cost.
- `N_JOBS = 1` keeps outer-fold execution serial. Increase it only if your machine has sufficient resources.
- `RUN_ABLATION = True` enables feature-group ablation.
- `RUN_LASSO_INTERPRETATION = True` enables coefficient inspection.

## Notebook outputs

The notebook creates `main/outfolder/` and writes the following files when their corresponding sections complete:

| File | Contents |
|---|---|
| `exp1_model_comparison.csv` | AUROC summaries by outcome, imputation method, and model, including bootstrap confidence intervals |
| `oof_predictions.npz` | Out-of-fold predictions and outcome labels used for ROC analysis |
| `exp2_ablation.csv` | Feature-group ablation results, if enabled |
| `lasso_coefficients.csv` | Non-zero fitted LASSO coefficients, if enabled |

The full nested-CV experiment suite may take a while because each outer fold performs an inner hyperparameter search. The notebook's code and paths can be checked independently of running every experiment; do not report performance values until the experiment has finished successfully.

## Optional script-based workflow

If the original `src/` scripts are present in the checkout and you want to run that separate synthetic-cohort workflow, use the commands below from the repository root:

```bash
python src/generate_data.py      # generates data/synthetic_cohort.csv
python src/train.py              # full script-based experiments
python src/train.py --quick      # quicker smoke test, if supported by the checked-out script
python src/demo.py               # demo, if included
python src/demo.py --index 3     # inspect one example, if supported by the checked-out script
```

These commands refer to the script-based workflow and are separate from the notebook workflow. Run only commands corresponding to files and options actually present in your repository.

## Methodological cautions

- Fit preprocessing and imputation within each training fold to avoid information leakage.
- Report AUROC together with uncertainty and class counts; small samples and imbalanced outcomes can make estimates unstable.
- LASSO coefficients describe associations in the fitted model and do not establish causality.
- Synthetic data and a small cleaned dataset do not establish real-world clinical validity. Independent external validation would be required before clinical use.

## Reference

Ostberg, N. and Peterson, D. *Predicting a Decline in Patient Reported Outcomes for Cancer Patients on Chemotherapy.* Stanford CS229, Spring 2020.
